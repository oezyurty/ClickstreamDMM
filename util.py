import argparse
import time
import os
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, config_enumerate
from pyro.optim import ClippedAdam

import pdb


class Emitter(nn.Module):
    """
    Parameterizes the Categorical observation likelihood `p(x_t | z_t)`
    """
    def __init__(self, input_dim, z_dim, emission_dim, use_feature_mask_emitter, min_x_scale):
        super().__init__()
        
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z_t, mini_batch_feature_mask_t=None):

        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = self.softmax(self.lin_hidden_to_input(h2))

        return ps




class Emitter_time(nn.Module):
    """
    Parameterizes the gaussian emission probability of time `p(t_t | z_t)`
    """
    def __init__(self, z_dim, emission_dim, min_x_scale):
        super().__init__()

        self.min_x_scale = min_x_scale
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_gate_hidden_to_t = nn.Linear(emission_dim, 1)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_proposed_mean_hidden_to_t = nn.Linear(emission_dim, 1)
        self.lin_sig = nn.Linear(1, 1)
        self.lin_z_to_loc = nn.Linear(z_dim, 1)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        #self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        #self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        """
        Given the latent `z_t` corresponding to the time step t
        we return the mean and scale vectors that parameterize the 
        gaussian distribution `p(t_t | z_t)`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t))
        gate = torch.sigmoid(self.lin_gate_hidden_to_t(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t))
        proposed_mean = self.lin_proposed_mean_hidden_to_t(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        scale = scale.add(self.min_x_scale)
        return loc, scale


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    """

    def __init__(self, z_dim, static_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.concat_dim = z_dim + static_dim
        self.lin_gate_z_to_hidden = nn.Linear(self.concat_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(self.concat_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, mini_batch_static):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        concat = torch.cat((z_t_1, mini_batch_static),dim=1)
        _gate = self.relu(self.lin_gate_z_to_hidden(concat))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(concat))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale

class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, s, x_{t:T}, t_{t:T})`
    """

    def __init__(self, z_dim, static_dim, rnn_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.concat_dim = z_dim + static_dim
        self.lin_z_to_hidden = nn.Linear(self.concat_dim , rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(2*rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(2*rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, mini_batch_static, h_rnn):
        # combine the rnn hidden state with a transformed version of z_t_1
        concat = torch.cat((z_t_1, mini_batch_static),dim=1)
        projected_concat = self.lin_z_to_hidden(concat)
        #h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(concat)) + h_rnn)
        h_combined = torch.cat((projected_concat, h_rnn),dim=1)

        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale

def transform_diff_t(diff_t):
    #Convert seconds to minutes, then take the log
    diff_t = diff_t/60 + 1
    diff_t = diff_t.log()
    return diff_t

class Predicter_Attention(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(y | z_T)`
    """
    def __init__(self, z_dim, att_dim, MLP_dims, time2vec_out=8, batch_first=True, use_cuda=True):
        super(Predicter_Attention, self).__init__() 
        self.z_dim = z_dim
        self.att_dim = att_dim
        self.MLP_dims = MLP_dims

        #Context vector is a parameter to measure the relevance of provided vector for y prediction
        bound = 1/np.sqrt(att_dim)
        self.context_vec = nn.Parameter(torch.zeros(att_dim, 1).uniform_(-bound, bound))
        #In attention framework, z_t's will be projected first
        self.projection_layer = nn.Linear(z_dim, att_dim)
        #There will be an activation function after projection
        self.tanh = nn.Tanh()
        #We use Beta Parameter to control sharpness/smoothness of Softmax function
        #self.Beta = torch.Tensor([0.1]).cuda()
        self.logBeta = torch.nn.Parameter(torch.Tensor([0]))
        #self.Beta = nn.Parameter(torch.ones(1))


        #We accepts MLP_dims as strings i.e. "48-24-12-..."
        #If MLP_dims is "-", it implies that there will be no middle layer
        if MLP_dims == "-":
            middle_layers = []
        else:
            middle_layers = MLP_dims.split("-")
        all_MLP_dimensions = [z_dim]
        for i in middle_layers:
            all_MLP_dimensions.append(int(i))
        #Last dim will be 1 for binary classification
        all_MLP_dimensions.append(1)
        self.lin_layers_nn = nn.ModuleList()
        for i in range(len(all_MLP_dimensions)-1):
            self.lin_layers_nn.append(nn.Linear(all_MLP_dimensions[i], all_MLP_dimensions[i+1]))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, cum_t, mini_batch_mask):
        cum_t = cum_t.squeeze(-1)
        diff_t = cum_t.max(dim=1, keepdim=True)[0] - cum_t #shape of (N,T)
        #Transform diff_t -> Convert to minute then take log
        transformed_diff_t = transform_diff_t(diff_t)

        #Note: z here has the shape of (N,T,z_dim)
        z_2d = z.reshape(-1, z.shape[2]) #shape of (N*T, z_dim)
        z_projected = self.tanh(self.projection_layer(z_2d)) #Now z_projected has the shape (N*T, att_dim)

        #Calculate weights (alpha) of z_projected
        #Below line is used originally (1)
        alpha = torch.mm(z_projected, self.context_vec) / np.sqrt(self.att_dim) #shape of alpha = (N*T,1)

        alpha = alpha.reshape(z.shape[0], z.shape[1]) #shape of alpha = (N,T)
        alpha = alpha - self.logBeta.exp() * transformed_diff_t
        alpha = alpha.masked_fill(mini_batch_mask == 0, -1e9)
        alpha = torch.softmax(alpha, dim=-1) #shape of alpha = (N,T)

        
        alpha = alpha.unsqueeze(-1).expand((alpha.shape + (z.shape[2],))) #shape of alpha = (N,T, z_dim)

        #Multiply z and alpha elementwise
        new_z = z * alpha #shape of new_z = (N,T, z_dim)
        #new_z = z_projected.reshape(z.shape[0], z.shape[1],-1) * alpha
        new_z = new_z.sum(axis=1) #shape of new_z = (N, z_dim)



        input_MLP = new_z
        for i in range(len(self.lin_layers_nn)-1):
            input_MLP = self.relu(self.lin_layers_nn[i](input_MLP))

        prob_out = self.sigmoid(self.lin_layers_nn[-1](input_MLP))
        return prob_out.flatten()

# this function takes a torch mini-batch and reverses each sequence
# (w.r.t. the temporal axis, i.e. axis=1).
def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch

def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0:seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask

def batchify(sequences, seq_lengths, sequences_feature_mask, static, y_sequence, y_mask_sequence, deltatime, cumtime, max_len=720, batch_size=128, use_feature_mask=True, cuda=True):

    keep_index = np.where(np.logical_and(seq_lengths <= max_len, seq_lengths > 0))[0]

    static = static[keep_index]
    sequences = sequences[keep_index]
    seq_lengths = seq_lengths[keep_index]
    deltatime = deltatime[keep_index]
    cumtime = cumtime[keep_index]
    if sequences_feature_mask is not None:
        sequences_feature_mask = sequences_feature_mask[keep_index]
    if y_sequence is not None:
        y_sequence = y_sequence[keep_index]
    if y_mask_sequence is not None:
        y_mask_sequence = y_mask_sequence[keep_index]


    N_data = len(seq_lengths)
    N_mini_batches = int(N_data / batch_size +
                            int(N_data % batch_size > 0))

    shuffled_indices = np.arange(N_data)

    batches = []

    for which_mini_batch in range(N_mini_batches):
        mini_batch_start = (which_mini_batch * batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * batch_size, N_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        
        batches.append(get_mini_batch(mini_batch_indices, sequences, seq_lengths, sequences_feature_mask, static, y_sequence, y_mask_sequence, deltatime, cumtime, keep_index, use_feature_mask=use_feature_mask, cuda=cuda))

    return batches

def get_mini_batch(mini_batch_indices, sequences, seq_lengths, sequences_feature_mask=None, static=None, y_sequence=None, y_mask_sequence=None, deltatime=None, cumtime=None, indices_dataset=None, use_feature_mask=False, cuda=False):
    # get the sequence lengths of the mini-batch
    #sorted_seq_lengths and sorted_mini_batch_indices
    seq_lengths = seq_lengths[mini_batch_indices]
    seq_lengths = torch.from_numpy(seq_lengths).type('torch.LongTensor')

    # sort the sequence lengths
    _, sorted_seq_length_indices = torch.sort(seq_lengths)
    sorted_seq_length_indices = sorted_seq_length_indices.flip(0)
    sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
    sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices.numpy()]

    # compute the length of the longest sequence in the mini-batch
    T_max = torch.max(seq_lengths)
    # this is the sorted mini-batch
    mini_batch = list(map(lambda x: torch.from_numpy(x[:T_max,:]).type('torch.DoubleTensor') , sequences[sorted_mini_batch_indices]))
    mini_batch = pad_sequence(mini_batch, batch_first=True).type('torch.DoubleTensor')
    #This is the sorted mini_batch_static
    mini_batch_static = static[sorted_mini_batch_indices]
    mini_batch_static = torch.from_numpy(mini_batch_static).type('torch.DoubleTensor')
    #this is the sorted deltatime
    mini_batch_deltatime = list(map(lambda x: torch.from_numpy(x[:T_max]).type('torch.DoubleTensor') , deltatime[sorted_mini_batch_indices]))
    mini_batch_deltatime = pad_sequence(mini_batch_deltatime, batch_first=True).type('torch.DoubleTensor')
    mini_batch_deltatime = mini_batch_deltatime.reshape(mini_batch_deltatime.shape + (1,)) #the final shape is N*T*1 

    #We need one shift to left for rnn input at guide. When there is no-shift, combiner and rnn gets the same time-step (previous one) for the current latent variable.
    #By shifting left, rnn gets the future delta_time instead. After shift, right side is padded by 0.
    mini_batch_deltatime_rnn = torch.nn.functional.pad(mini_batch_deltatime, (0,0,0,1), "constant", 0)[:,1:,:]
    #Why (0,0,0,1)? -> Because no padding for last dimension, and we pad the right a 2nd to last dimension. 

    #this is the sorted cumtime
    mini_batch_cumtime = list(map(lambda x: torch.from_numpy(x[:T_max]).type('torch.DoubleTensor') , cumtime[sorted_mini_batch_indices]))
    mini_batch_cumtime = pad_sequence(mini_batch_cumtime, batch_first=True).type('torch.DoubleTensor')
    mini_batch_cumtime = mini_batch_cumtime.reshape(mini_batch_cumtime.shape + (1,)) #the final shape is N*T*1 

    # get mask for mini-batch
    mini_batch_mask = get_mini_batch_mask(mini_batch, sorted_seq_lengths)
    #get the y values (mortality labels) of mini-batch
    if y_sequence is None:
        y_mini_batch = None
    else:
        y_mini_batch = y_sequence[sorted_mini_batch_indices]
        y_mini_batch = torch.from_numpy(y_mini_batch).type('torch.DoubleTensor')

    #get y mask values (for semi-supervised learning)
    if y_mask_sequence is None:
        y_mask_mini_batch = None
    else:
        y_mask_mini_batch = y_mask_sequence[sorted_mini_batch_indices]
        y_mask_mini_batch = torch.from_numpy(y_mask_mini_batch).type('torch.DoubleTensor')


    #Below line was just for convenience
    mini_batch_feature_mask = None
    #mini_batch_reversed_with_mask = reverse_sequences(torch.cat((mini_batch, mini_batch_deltatime),dim=-1), sorted_seq_lengths)
    mini_batch_reversed_with_mask = reverse_sequences(torch.cat((mini_batch, mini_batch_deltatime_rnn),dim=-1), sorted_seq_lengths)

    # cuda() here because need to cuda() before packing
    if cuda:
        mini_batch = mini_batch.cuda()
        mini_batch_static = mini_batch_static.cuda()
        mini_batch_mask = mini_batch_mask.cuda()
        mini_batch_deltatime = mini_batch_deltatime.cuda()
        mini_batch_cumtime = mini_batch_cumtime.cuda()
        if y_mini_batch is not None:
            y_mini_batch = y_mini_batch.cuda()
        if y_mask_mini_batch is not None:
            y_mask_mini_batch = y_mask_mini_batch.cuda()
        mini_batch_reversed_with_mask = mini_batch_reversed_with_mask.cuda()
        if mini_batch_feature_mask is not None:
            mini_batch_feature_mask = mini_batch_feature_mask.cuda()
        
    # do sequence packing
    mini_batch_reversed_with_mask = nn.utils.rnn.pack_padded_sequence(mini_batch_reversed_with_mask,
                                                            sorted_seq_lengths,
                                                            batch_first=True)

    return mini_batch_static, mini_batch, mini_batch_reversed_with_mask, mini_batch_mask, sorted_seq_lengths, mini_batch_feature_mask, y_mini_batch, y_mask_mini_batch, mini_batch_deltatime, mini_batch_cumtime, indices_dataset[sorted_mini_batch_indices]



def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output
