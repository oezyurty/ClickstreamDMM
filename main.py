import argparse
import time
import os
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from pyro.infer import Predictive

from util import Emitter, Emitter_time, GatedTransition, Combiner, Predicter_Attention
from util import batchify, reverse_sequences, get_mini_batch, pad_and_reverse

import logging
import json
from collections import namedtuple

#You may comment out this later on
pyro.set_rng_seed(1234)

#just want to suppres warning message for dmm.predicter.Beta
import warnings
warnings.filterwarnings("ignore", message="predicter.Beta was not registered in the param store")

def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log

class DMM(nn.Module):

    def __init__(self, input_dim=12, z_dim=8, static_dim=5, min_x_scale=0.2, emission_dim=16,
                 transition_dim=32, linear_gain=False, att_dim=16, time2vec_out = 8, MLP_dims="12-3", guide_GRU=False, rnn_dim=16, rnn_num_layers=1, class_weights=(1,1), rnn_dropout_rate=0.0,
                 num_iafs=0, iaf_dim=50, use_feature_mask=False, use_cuda=False):
        super().__init__()
        #Keep the flag of use_feature_mask
        self.use_feature_mask = use_feature_mask
        # instantiate PyTorch modules used in the model and guide below
        # For now, we limit that emitter will never use feature mask
        self.emitter = Emitter(input_dim, z_dim, emission_dim, False, min_x_scale)
        self.emitter_time = Emitter_time(z_dim, transition_dim, min_x_scale)
        self.trans = GatedTransition(z_dim, static_dim, transition_dim)
        self.combiner = Combiner(z_dim, static_dim+1, rnn_dim)
        self.predicter = Predicter_Attention(z_dim, att_dim, MLP_dims, time2vec_out, batch_first=True, use_cuda=use_cuda)

        self.linear_gain = linear_gain
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if rnn_num_layers == 1 else rnn_dropout_rate
        #Below, +1 for input_size comes from time(t)
        if not guide_GRU:
            self.rnn = nn.RNN(input_size=input_dim + use_feature_mask*input_dim + 1, hidden_size=rnn_dim, nonlinearity='relu',
                                batch_first=True, bidirectional=False, num_layers=rnn_num_layers,
                                dropout=rnn_dropout_rate)
        else:
            self.rnn = nn.GRU(input_size=input_dim + use_feature_mask*input_dim + 1, hidden_size=rnn_dim, 
                                batch_first=True, bidirectional=False, num_layers=rnn_num_layers,
                                dropout=rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.class_weights = class_weights

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # HERE WE DEFINE model AND guide functions

    def model(self, mini_batch_static, mini_batch, mini_batch_reversed, mini_batch_mask,
                mini_batch_seq_lengths, mini_batch_feature_mask=None, y_mini_batch=None, y_mask_mini_batch=None, mini_batch_deltatime=None, mini_batch_cumtime=None, annealing_factor=1.0, regularizer=1.0, renormalize_y_weight=False):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        #Inıtialize the empty list to keep hidden states of every time step
        all_hidden_states = []

        for t in pyro.markov(range(1, T_max + 1)):

            with pyro.plate("z_minibatch_"+str(t), len(mini_batch), dim=-1):

                z_loc, z_scale = self.trans(z_prev, mini_batch_static)

                with poutine.scale(scale=annealing_factor*regularizer/T_max):
                    z_t = pyro.sample("z_%d" % t,
                                        dist.Normal(z_loc, z_scale)
                                            .mask(mini_batch_mask[:, t - 1:t])
                                            .to_event(1))

                #Time emission wont be done for the last time step
                if t < T_max:
                    t_loc, t_scale = self.emitter_time(z_t)

                    with poutine.scale(scale=10*regularizer/(T_max-1)):
                        pyro.sample("obs_t_%d" % t,
                                        dist.Normal(t_loc, t_scale)
                                            .mask(mini_batch_mask[:, t: t+1])
                                            .to_event(1),
                                        obs=mini_batch_deltatime[:, t, :])

                # compute the probabilities that parameterize the OneHotCategorical likelihood
                emission_probs_t  = self.emitter(z_t)

                # the next statement instructs pyro to observe x_t according to the
                # OneHotCategorical distribution p(x_t|z_t)

                with poutine.scale(scale=10*regularizer/T_max):
                    pyro.sample("obs_x_%d" % t,
                                    dist.OneHotCategorical(emission_probs_t)
                                        .mask(mini_batch_mask[:, t - 1]),
                                    obs=mini_batch[:, t - 1, :])
                

            # the latent sampled at this time step will be conditioned upon
            # in the next time step so keep track of it
            z_prev = z_t

            #Add hidden states at time t to all_hidden_states
            all_hidden_states.append(z_prev)

        #The predicter (Attention+MLP) will use all the hidden states {z_t}'s to make prediction
        all_hidden_states = torch.stack(all_hidden_states).transpose(0,1)


        #The last time step's z_t, which is z_prev, will be used to predict the mortality label
        purchase_probs = self.predicter(all_hidden_states, mini_batch_cumtime, mini_batch_mask)
        #Adjust the scaling weights for each class such that 
        #Total weight of y's will be => num_data_points
        if y_mini_batch is not None:
            weights = torch.zeros_like(y_mini_batch)
            weights[y_mini_batch == 0] = self.class_weights[0]
            weights[y_mini_batch == 1] = self.class_weights[1]  
        else:
            weights = 1
        with pyro.plate("y_minibatch", len(mini_batch)):
            poutine_y_mask = torch.ones((len(mini_batch))).cuda() == 1 if y_mask_mini_batch is None else y_mask_mini_batch == 1
            #Re-normalize weights to cover missing y values
            if y_mask_mini_batch is not None and renormalize_y_weight:
                weights = weights/y_mask_mini_batch.mean()
            with poutine.mask(mask=poutine_y_mask): 
                with poutine.scale(scale=weights):
                    pyro.sample("y", dist.Bernoulli(purchase_probs), obs=y_mini_batch)

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, mini_batch_static, mini_batch, mini_batch_reversed, mini_batch_mask,
                mini_batch_seq_lengths, mini_batch_feature_mask=None, y_mini_batch=None, y_mask_mini_batch=None, mini_batch_deltatime=None, mini_batch_cumtime=None, annealing_factor=1.0, regularizer=1.0, renormalize_y_weight=False):
        
        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        #h_0_contig = h_0_contig.type('torch.DoubleTensor')
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch), dim=-1):
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, torch.cat((mini_batch_static, mini_batch_deltatime[:,t-1,:]), dim=1), rnn_output[:, t - 1, :])

                if len(self.iafs) > 0:
                    z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                    assert z_dist.event_shape == (self.z_q_0.size(0),)
                    assert z_dist.batch_shape[-1:] == (len(mini_batch),)
                else:
                    z_dist = dist.Normal(z_loc, z_scale)
                    assert z_dist.event_shape == ()
                    assert z_dist.batch_shape[-2:] == (len(mini_batch), self.z_q_0.size(0))

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor*regularizer/T_max):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t = pyro.sample("z_%d" % t,
                                            z_dist.mask(mini_batch_mask[:, t - 1]))
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t = pyro.sample("z_%d" % t,
                                            z_dist.mask(mini_batch_mask[:, t - 1:t])
                                            .to_event(1))
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t


def main(args):

    # setup logging
    if not args.eval_mode:
        log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, args.log))
    else:
        log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, "eval_"+args.log))
    log(args)

    #log(parser.get_default('z_dim'))

    #If in training mode, save the argparse arguments for eval mode to be used later on
    if not args.eval_mode:
        if args.load_model == '':
            with open(os.path.join(args.experiments_main_folder, args.experiment_folder,'commandline_args.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        else:
            pretrained_model_path = '/'.join(args.load_model.split('/')[:-1])
            with open(os.path.join(args.experiments_main_folder, args.experiment_folder, pretrained_model_path, 'commandline_args.txt'), 'r') as f:
                prev_args_dict_ = json.load(f)
            curr_args = {}
            for attr, value in prev_args_dict_.items():
                curr_args[attr] = value
            # filter_label will be forced to be False for our current setting (It can be overwritten by current args)
            curr_args['filter_label'] = False
            curr_args['filter_no_label'] = False
            #Now overwrite curr_args with args
            for attr, value in args.__dict__.items():
                if not attr in curr_args.keys() or value != parser.get_default(attr):
                    curr_args[attr] = value
            #Now we can replace args with curr_args
            args = namedtuple("Args", curr_args.keys())(*curr_args.values())
            #Save the current args to 'commandline_args.txt'
            with open(os.path.join(args.experiments_main_folder, args.experiment_folder,'commandline_args.txt'), 'w') as f:
                json.dump(curr_args, f, indent=2)

            #For Debug
            log("Last situtation of args after mergin previous and current settings:")
            log(args)


    #If in eval mode, load the previously saved argparse arguments to load all models correctly
    else:
        with open(os.path.join(args.experiments_main_folder, args.experiment_folder,'commandline_args.txt'), 'r') as f:
            saved_args_dict_ = json.load(f)
        #If the model was saved before the latest updates, some arguments can be missing in saved_args. We fill them with default arguments
        for attr, value in args.__dict__.items():
            if not attr in saved_args_dict_.keys():
                saved_args_dict_[attr] = value
        saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    data_folder = args.data_folder

    #Choose static features to be added to DMM model
    training_static = np.load(os.path.join(data_folder, 'static_train.npy'))

    val_static = np.load(os.path.join(data_folder, 'static_val.npy'))

    test_static = np.load(os.path.join(data_folder, 'static_test.npy'))

    #Load delta_time_log
    training_deltatime = np.load(os.path.join(data_folder, 'delta_time_train_log.npy'), allow_pickle=True)

    val_deltatime = np.load(os.path.join(data_folder, 'delta_time_val_log.npy'), allow_pickle=True)

    test_deltatime = np.load(os.path.join(data_folder, 'delta_time_test_log.npy'), allow_pickle=True)

    #Load cum time
    training_cumtime = np.load(os.path.join(data_folder, 'cum_time_train.npy'), allow_pickle=True)

    val_cumtime = np.load(os.path.join(data_folder, 'cum_time_val.npy'), allow_pickle=True)

    test_cumtime = np.load(os.path.join(data_folder, 'cum_time_test.npy'), allow_pickle=True)


    training_data_sequences = np.load(os.path.join(data_folder, 'timeseries_train.npy'), allow_pickle=True)

    training_seq_lengths = np.array(list(map(lambda x: len(x), training_data_sequences)))

    val_data_sequences = np.load(os.path.join(data_folder, 'timeseries_val.npy'), allow_pickle=True)

    val_seq_lengths = np.array(list(map(lambda x: len(x), val_data_sequences)))

    test_data_sequences = np.load(os.path.join(data_folder, 'timeseries_test.npy'), allow_pickle=True)

    test_seq_lengths = np.array(list(map(lambda x: len(x), test_data_sequences)))

    #Load the purchase labels for each split
    y_train = np.load(os.path.join(data_folder, "y_train.npy"))
    y_train = y_train.flatten()
    y_val = np.load(os.path.join(data_folder, "y_val.npy"))
    y_val = y_val.flatten()
    y_test = np.load(os.path.join(data_folder, "y_test.npy"))
    y_test = y_test.flatten()

    #Calculate the weights of each class for weighted loss
    #Assumption! Minority is class 1. Otherwise, revise the below code
    flag_weighted_loss = (not args.eval_mode and args.weighted_loss) or (args.eval_mode and saved_args.weighted_loss)
    log(flag_weighted_loss)
    class_weights = (1,1)
    if flag_weighted_loss:
        prop = 1 - y_val.mean()
        weight_0 = 1 / (2 * prop)
        weight_1 = weight_0 * prop / (1 - prop)
        class_weights = (weight_0, weight_1)
        log("Loss is weighted with " + str(class_weights))

    #BELOW BLOCK BELONGS TO A LEGACY VERSION
    #Now load the feature-level maskings to be added to the model if flag_load_feature_mask
    #Idea of flag:
    #If feature mask will be used for "emitter and guide" or "ELBO loss masking", we will load them.
    #Current Approach: 
        #If only args.use_feature_mask_ELBO : mask will only be used during ELBO computation
        #If args.use_feature_mask: It will be used for both "emitter and guide" and "ELBO computation" (Basically we don't check args.use_feature_mask_ELBO at all. Still make it true for convenience!)
    flag_load_feature_mask = (not args.eval_mode and (args.use_feature_mask or args.use_feature_mask_ELBO)) or (args.eval_mode and (saved_args.use_feature_mask or saved_args.use_feature_mask_ELBO))
    print(flag_load_feature_mask)
    if flag_load_feature_mask:
        training_data_sequences_mask = np.load(os.path.join(data_folder, 'time_series_training_masking.npy'), allow_pickle=True)

        val_data_sequences_mask = np.load(os.path.join(data_folder, 'time_series_val_masking.npy'), allow_pickle=True)

        test_data_sequences_mask = np.load(os.path.join(data_folder, 'time_series_test_masking.npy'), allow_pickle=True)

        #EXPERIMENTAL PART: Convert all imputed values to -4 (or something out of current range)
        if (not args.eval_mode and args.convert_imputations) or (args.eval_mode and saved_args.convert_imputations):
            training_data_sequences[training_data_sequences_mask == 0] = -4
            val_data_sequences[val_data_sequences_mask == 0] = -4
            test_data_sequences[test_data_sequences_mask == 0] = -4
    else:
        training_data_sequences_mask = None
        val_data_sequences_mask = None
        test_data_sequences_mask = None

    #Load the label masking if provided (for SEMI-SUPERVISED experiments)
    flag_load_label_mask = (not args.eval_mode and args.label_masking != '') or (args.eval_mode and saved_args.label_masking != '')
    if flag_load_label_mask:
        mask_name = args.label_masking if not args.eval_mode else saved_args.label_masking
        y_train_mask = np.load(os.path.join(data_folder, "y_training_masking_"+mask_name+".npy"))
    else:
        y_train_mask = None
    #We won't use any mask for val or test at all!
    y_val_mask = None
    y_test_mask = None

    #FOR SEMI-SUPERVISED EXPERIMENTS
    #If filter_no_label is on, filter out all the training points that are masked out by the label
    #To make filter_no_label flag on: filter_no_label must be true and a valid label_masking must be entered
    #Else if filter_label is on, filter out all the training points having the label (Basically, DMM works like autoencoder (i.e. unsupervised))
    #To make filter_label flag on: filter_label must be true and a valid label_masking must be entered

    flag_filter_no_label = flag_load_label_mask and ((not args.eval_mode and args.filter_no_label) or (args.eval_mode and saved_args.filter_no_label)) 
    flag_filter_label = flag_load_label_mask and ((not args.eval_mode and args.filter_label) or (args.eval_mode and saved_args.filter_label)) 

    if flag_filter_no_label:
        training_static = training_static[y_train_mask == 1]
        training_data_sequences = training_data_sequences[y_train_mask == 1]
        training_seq_lengths = training_seq_lengths[y_train_mask == 1]
        y_train = y_train[y_train_mask == 1]
        if training_data_sequences_mask is not None:
            training_data_sequences_mask = training_data_sequences_mask[y_train_mask == 1]
        y_train_mask = y_train_mask[y_train_mask == 1]

    elif flag_filter_label:
        training_static = training_static[y_train_mask == 0]
        training_data_sequences = training_data_sequences[y_train_mask == 0]
        training_seq_lengths = training_seq_lengths[y_train_mask == 0]
        y_train = y_train[y_train_mask == 0]
        if training_data_sequences_mask is not None:
            training_data_sequences_mask = training_data_sequences_mask[y_train_mask == 0]
        y_train_mask = y_train_mask[y_train_mask == 0]
        #Additional to filter_no_label, mask out all the y_val and y_test to assure consistent validation and test losses
        y_val_mask = torch.zeros_like(y_val)
        y_test_mask = torch.zeros_like(y_test)

    #Get the batches of the sessions
    use_feature_mask_mini_batch = (not args.eval_mode and args.use_feature_mask) or (args.eval_mode and saved_args.use_feature_mask)
    batches_training = batchify(training_data_sequences, training_seq_lengths, training_data_sequences_mask, training_static, y_train, y_train_mask, training_deltatime, training_cumtime, max_len=360, batch_size=args.mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)
    batches_val = batchify(val_data_sequences, val_seq_lengths, val_data_sequences_mask, val_static, y_val, y_val_mask, val_deltatime, val_cumtime, max_len=360, batch_size=args.mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)
    batches_test = batchify(test_data_sequences, test_seq_lengths, test_data_sequences_mask, test_static, y_test, y_test_mask, test_deltatime, test_cumtime, max_len=360, batch_size=args.mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)


    N_train_data = args.mini_batch_size * (len(batches_training)-1) + len(batches_training[-1][0])
    N_train_time_slices = float(np.sum(training_seq_lengths))
    N_mini_batches = len(batches_training)

    log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
        (N_train_data, training_seq_lengths.mean(), N_mini_batches))

    # how often we do validation/test evaluation during training
    val_test_frequency = args.eval_freq
    # the number of samples we use to do the evaluation
    n_eval_samples = 1


    # instantiate the dmm (if eval_mode, then use the previous setting)
    # Same idea to setup optimizer (if eval_mode, then use the previous setting)
    if not args.eval_mode:
        dmm = DMM(input_dim=batches_training[0][1].shape[2], static_dim=batches_training[0][0].shape[1], z_dim=args.z_dim, min_x_scale=args.min_x_scale, emission_dim=args.emission_dim, transition_dim=args.transition_dim, linear_gain=args.linear_gain, att_dim = args.att_dim, time2vec_out=args.time2vec_out, MLP_dims=args.MLP_dims,
                    guide_GRU=args.guide_GRU, rnn_dropout_rate=args.rnn_dropout_rate, rnn_dim=args.rnn_dim, rnn_num_layers=args.rnn_num_layers, class_weights=class_weights,
                    num_iafs=args.num_iafs, iaf_dim=args.iaf_dim, use_feature_mask=args.use_feature_mask, use_cuda=args.cuda)
        adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                        "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                        "weight_decay": args.weight_decay}
    else:
        dmm = DMM(input_dim=batches_training[0][1].shape[2], static_dim=batches_training[0][0].shape[1], z_dim=saved_args.z_dim, min_x_scale=saved_args.min_x_scale, emission_dim=saved_args.emission_dim, transition_dim=saved_args.transition_dim, linear_gain=saved_args.linear_gain, att_dim = saved_args.att_dim, time2vec_out=saved_args.time2vec_out, MLP_dims=saved_args.MLP_dims,
                    guide_GRU=saved_args.guide_GRU, rnn_dropout_rate=saved_args.rnn_dropout_rate, rnn_dim=saved_args.rnn_dim, rnn_num_layers=saved_args.rnn_num_layers, class_weights=class_weights,
                    num_iafs=saved_args.num_iafs, iaf_dim=saved_args.iaf_dim, use_feature_mask=saved_args.use_feature_mask, use_cuda=saved_args.cuda)
        adam_params = {"lr": saved_args.learning_rate, "betas": (saved_args.beta1, saved_args.beta2),
                        "clip_norm": saved_args.clip_norm, "lrd": saved_args.lr_decay,
                        "weight_decay": saved_args.weight_decay}

    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    if args.tmc:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC")
        tmc_loss = TraceTMC_ELBO()
        dmm_guide = config_enumerate(dmm.guide, default="parallel", num_samples=args.tmc_num_samples, expand=False)
        svi = SVI(dmm.model, dmm_guide, adam, loss=tmc_loss)
    elif args.tmcelbo:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC ELBO")
        elbo = TraceEnum_ELBO()
        dmm_guide = config_enumerate(dmm.guide, default="parallel", num_samples=args.tmc_num_samples, expand=False)
        svi = SVI(dmm.model, dmm_guide, adam, loss=elbo)
    else:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint(is_best=False):
        save_model = os.path.join(args.experiments_main_folder, args.experiment_folder, args.save_model)
        save_opt = os.path.join(args.experiments_main_folder, args.experiment_folder, args.save_opt)
        save_param = os.path.join(args.experiments_main_folder, args.experiment_folder, "mymodelparams.pt")
        if is_best:
            save_model+='_best'
            save_opt+='_best'
            save_param = os.path.join(args.experiments_main_folder, args.experiment_folder, "mymodelparams_best.pt")
        log("saving model to %s..." % save_model)
        torch.save(dmm.state_dict(), save_model)
        log("saving optimizer states to %s..." % save_opt)
        adam.save(save_opt)
        #log("saving param_store to %s..." % save_param)
        #pyro.get_param_store().save(save_param)
        log("done saving model and optimizer checkpoints to disk.")

    # loads the model and optimizer states from disk
    def load_checkpoint():
        load_model = os.path.join(args.experiments_main_folder, args.experiment_folder, args.load_model)
        load_opt = os.path.join(args.experiments_main_folder, args.experiment_folder, args.load_opt)
        load_param = os.path.join(args.experiments_main_folder, args.experiment_folder, "mymodelparams.pt")
        if "best" in args.load_model:
            load_param = os.path.join(args.experiments_main_folder, args.experiment_folder, "mymodelparams_best.pt")
        
        
        assert exists(load_opt) and exists(load_model), \
            "--load-model and/or --load-opt misspecified"
        log("loading model from %s..." % load_model)
        #Yilmazcan Deneme: For now block the below part and see what load_param does
        #pyro.clear_param_store()
        dmm.load_state_dict(torch.load(load_model))
        #dmm = torch.load(load_model)
        
        #if model is loaded in training mode, randomly initialize the weights of predicter 
        #so that it will be ready for supervised learning after dmm being trained with unsupervised methods
        #If this step is not done, predicter starts with weigths quite close to 0, and cannot learn anything from it
        #SPECIAL NOTE: Adam won't be loaded for below scenario, instead it's created by current args (so that we can change learning rate etc.)
        if not args.eval_mode and args.filter_no_label:
            for i in range(len(dmm.predicter.lin_layers_nn)):
                print("Re-initalization will be done for Linear layer " + str(i))
                activation = 'sigmoid' if i == len(dmm.predicter.lin_layers_nn)-1 else 'relu'
                nn.init.xavier_uniform_(dmm.predicter.lin_layers_nn[i].weight.data, gain=nn.init.calculate_gain(activation))
                #nn.init.uniform_(dmm.predicter.lin_layers_nn[i].weight.data, a=-0.1, b=0.1)
                nn.init.zeros_(dmm.predicter.lin_layers_nn[i].bias.data)
                #print(dmm.predicter.lin_layers_nn[i].weight.data)
                #print(dmm.predicter.lin_layers_nn[i].bias.data)
            log("Predicter weights are re-initialized!")
            log("Adam optimizer will not be loaded from previous model, instead it's created by current args")
        else:
            log("loading optimizer states from %s..." % load_opt)
            adam.load(load_opt)
        log("done loading model and optimizer states.")
        
    # prepare a mini-batch and take a gradient step to minimize -elbo
    def process_minibatch(epoch, which_mini_batch):
        dmm.rnn.train()
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = args.minimum_annealing_factor
            max_af = args.maximum_annealing_factor
            annealing_factor = min_af + (max_af - min_af) * \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                    float(args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = args.maximum_annealing_factor

        #Calculate if renormalize_y_weight will be done for svi.step
        renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)

        # grab a fully prepped mini-batch using the helper function in the data loader
        mini_batch_static, mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths, mini_batch_feature_mask, y_mini_batch, y_mask_mini_batch, mini_batch_deltatime, mini_batch_cumtime, _ = batches_training[which_mini_batch]
        
        # do an actual gradient step
        loss = svi.step(mini_batch_static,mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, mini_batch_feature_mask, y_mini_batch, y_mask_mini_batch, mini_batch_deltatime, mini_batch_cumtime, annealing_factor, args.regularizer, renormalize_y_weight)

        return loss

    # helper function for doing evaluation
    def do_evaluation():
        '''
        Do also evaulation batch by batch (otherwise we have GPU memory problem)
        '''
        dmm.rnn.eval()

        #During the evaluation, we will always use maximum annealing factor
        annealing_factor = args.maximum_annealing_factor
        #Calculate regularizer depending on which mode we are running (train or eval mode)
        regularizer = args.regularizer if not args.eval_mode else saved_args.regularizer
        #Calculate if renormalize_y_weight will be done for svi.evaluate_loss
        renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)


        #EVALUATION FOR VALIDATION SET
        N_val_data = args.mini_batch_size * (len(batches_val)-1) + len(batches_val[-1][0])
        N_val_time_slices = float(np.sum(val_seq_lengths))
        eval_N_mini_batches = len(batches_val)

        val_nll = 0
        for i in range(eval_N_mini_batches):

            val_batch_static, val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, val_batch_deltatime, val_batch_cumtime, _ = batches_val[i]

            val_nll_batch = svi.evaluate_loss(val_batch_static, val_batch, val_batch_reversed, val_batch_mask,
                                    val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, val_batch_deltatime, val_batch_cumtime, annealing_factor, regularizer, renormalize_y_weight)
            val_nll+=val_nll_batch

        val_nll = val_nll/float(N_val_data)

        #EVALUATION FOR TEST SET
        N_test_data = args.mini_batch_size * (len(batches_test)-1) + len(batches_test[-1][0])
        N_test_time_slices = float(np.sum(test_seq_lengths))
        eval_N_mini_batches = len(batches_test)

        test_nll = 0
        for i in range(eval_N_mini_batches):

            test_batch_static, test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, test_batch_deltatime, test_batch_cumtime, _ = batches_test[i]

            test_nll_batch = svi.evaluate_loss(test_batch_static, test_batch, test_batch_reversed, test_batch_mask,
                                        test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, test_batch_deltatime, test_batch_cumtime, annealing_factor, regularizer, renormalize_y_weight)
            test_nll+=test_nll_batch


        test_nll = test_nll/float(N_test_data)

        dmm.rnn.train()

        return val_nll, test_nll

    def do_evaluation_rocauc(mini_batch_size=args.mini_batch_size, num_samples=10, verbose=False):
        '''
        Do also evaulation batch by batch (otherwise we have GPU memory problem)
        additional_mop -> including the mean and std of prediction for different samples of num_samples
        '''
        dmm.rnn.eval()

        #During the evaluation, we will always use maximum annealing factor
        annealing_factor = args.maximum_annealing_factor
        #Calculate regularizer depending on which mode we are running (train or eval mode)
        regularizer = args.regularizer if not args.eval_mode else saved_args.regularizer
        #Calculate if renormalize_y_weight will be done for pred
        renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)

        #Initialize the predicter module
        pred = Predictive(model=dmm.model, guide=dmm.guide, num_samples=num_samples)


        #EVALUATION FOR VALIDATION SET
        N_val_data = args.mini_batch_size * (len(batches_val)-1) + len(batches_val[-1][0])
        N_val_time_slices = float(np.sum(val_seq_lengths))
        eval_N_mini_batches = len(batches_val)

        all_y_val = []
        all_purchase_probs_val = []
        for i in range(eval_N_mini_batches):
            val_batch_static, val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, val_batch_deltatime, val_batch_cumtime, _ = batches_val[i]

            pred_dict_val = pred(val_batch_static, val_batch, val_batch_reversed, val_batch_mask,
                                    val_batch_seq_lengths, val_batch_feature_mask, None, None, val_batch_deltatime, val_batch_cumtime, annealing_factor, regularizer, renormalize_y_weight)

            #Do Accuracy and ROCAUC analysis for mortality prediction
            list_purchase_probs = []
            z_name = 'z_'
            for j in range(num_samples):
                all_z_vals = []
                # I added [0] here since the original dim was (1,6379,8)
                for w in range(1, val_batch.shape[1]+1):
                    z_i_name = z_name + str(w)
                    z_i_val = pred_dict_val[z_i_name][j]
                    all_z_vals.append(z_i_val)
                all_z_vals = torch.stack(all_z_vals).transpose(0,1)
                purchase_probs = dmm.predicter(all_z_vals, val_batch_cumtime, val_batch_mask)
                list_purchase_probs.append(purchase_probs.detach().cpu().numpy())

            purchase_probs = np.mean(np.array(list_purchase_probs), axis=0)
            all_purchase_probs_val = all_purchase_probs_val + list(purchase_probs)


            all_y_val = all_y_val + list(y_val_batch.detach().cpu().numpy())

            if verbose:
                print("In Validation split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

        if verbose:
            print("Validation Done!                   ")

        roc_auc_val = roc_auc_score(np.array(all_y_val), np.array(all_purchase_probs_val))

        #EVALUATION FOR TEST SET
        N_test_data = args.mini_batch_size * (len(batches_test)-1) + len(batches_test[-1][0])
        N_test_time_slices = float(np.sum(test_seq_lengths))
        eval_N_mini_batches = len(batches_test)

        all_y_test = []
        all_purchase_probs_test = []
        all_purchase_probs_test_std = []
        all_test_batch_indices = []
        for i in range(eval_N_mini_batches):
            test_batch_static, test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, test_batch_deltatime, test_batch_cumtime, index_test_batch = batches_test[i]

            all_test_batch_indices = all_test_batch_indices + list(index_test_batch)

            pred_dict_test = pred(test_batch_static, test_batch, test_batch_reversed, test_batch_mask,
                                        test_batch_seq_lengths, test_batch_feature_mask, None, None, test_batch_deltatime, test_batch_cumtime, annealing_factor, regularizer, renormalize_y_weight)
            
            #Do Accuracy and ROCAUC analysis for mortality prediction
            list_purchase_probs = []
            z_name = 'z_'
            for j in range(num_samples):
                all_z_tests = []
                for w in range(1, test_batch.shape[1]+1):
                    z_i_name = z_name + str(w)
                    z_i_test = pred_dict_test[z_i_name][j]
                    all_z_tests.append(z_i_test)
                all_z_tests = torch.stack(all_z_tests).transpose(0,1)
                purchase_probs = dmm.predicter(all_z_tests, test_batch_cumtime, test_batch_mask)
                list_purchase_probs.append(purchase_probs.detach().cpu().numpy())

            purchase_probs = np.mean(np.array(list_purchase_probs), axis=0)
            purchase_probs_std = np.std(np.array(list_purchase_probs), axis=0)

            all_purchase_probs_test = all_purchase_probs_test + list(purchase_probs)
            all_purchase_probs_test_std = all_purchase_probs_test_std + list(purchase_probs_std)


            all_y_test = all_y_test + list(y_test_batch.detach().cpu().numpy())
            
            if verbose:
                print("In Test split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

        if verbose:
            print("Test Done!                    ")

        roc_auc_test = roc_auc_score(np.array(all_y_test), np.array(all_purchase_probs_test))

        dmm.rnn.train()

        return roc_auc_val, roc_auc_test, np.array(all_y_test), np.array(all_purchase_probs_test), np.array(all_purchase_probs_test_std), np.array(all_test_batch_indices)


    def do_evaluation_rocauc_custom_time(mini_batch_size=args.mini_batch_size, num_samples=10, cropped_t_from_last=1, do_eval_for_val=True ,verbose=False):
        '''
        Do also evaulation batch by batch (otherwise we have GPU memory problem)
        '''
        dmm.rnn.eval()

        #Calculate if feature mask is used for get_mini_batch(...):
        use_feature_mask_mini_batch = (not args.eval_mode and args.use_feature_mask) or (args.eval_mode and saved_args.use_feature_mask)
        #During the evaluation, we will always use maximum annealing factor
        annealing_factor = args.maximum_annealing_factor
        #Calculate regularizer depending on which mode we are running (train or eval mode)
        regularizer = args.regularizer if not args.eval_mode else saved_args.regularizer
        #Calculate if renormalize_y_weight will be done for pred
        renormalize_y_weight = (not args.eval_mode and args.renormalize_y_weight) or (args.eval_mode and saved_args.renormalize_y_weight)

        #Initialize the predicter module
        pred = Predictive(model=dmm.model, guide=dmm.guide, num_samples=num_samples)

        if do_eval_for_val:

            #MODIFY VALIDATION DATA TO CROP LATEST t TIME STEPS
            custom_val_data_sequences = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , val_data_sequences)))
            custom_val_deltatime = np.array(list(map(lambda x: x[:-cropped_t_from_last] , val_deltatime)))
            custom_val_cumtime = np.array(list(map(lambda x: x[:-cropped_t_from_last] , val_cumtime)))

            
            custom_val_seq_lengths = val_seq_lengths - cropped_t_from_last

            if val_data_sequences_mask is not None:
                custom_val_data_sequences_mask = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , val_data_sequences_mask)))
            else:
                custom_val_data_sequences_mask = None


            batches_val = batchify(custom_val_data_sequences, custom_val_seq_lengths, custom_val_data_sequences_mask, val_static, y_val, y_val_mask, custom_val_deltatime, custom_val_cumtime, max_len=360 - cropped_t_from_last, batch_size=mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)
            

            #EVALUATION FOR VALIDATION SET
            N_val_data = mini_batch_size * (len(batches_val)-1) + len(batches_val[-1][0])
            N_val_time_slices = float(np.sum(custom_val_seq_lengths))
            eval_N_mini_batches = len(batches_val)

            all_y_val = []
            all_purchase_probs_val = []
            for i in range(eval_N_mini_batches):
                val_batch_static, val_batch, val_batch_reversed, val_batch_mask, val_batch_seq_lengths, val_batch_feature_mask, y_val_batch, y_mask_val_batch, val_batch_deltatime, val_batch_cumtime, _ = batches_val[i]

                pred_dict_val = pred(val_batch_static, val_batch, val_batch_reversed, val_batch_mask,
                                        val_batch_seq_lengths, val_batch_feature_mask, None, None, val_batch_deltatime, val_batch_cumtime, annealing_factor, regularizer, renormalize_y_weight)

                #Do Accuracy and ROCAUC analysis for mortality prediction
                list_purchase_probs = []
                z_name = 'z_'
                for j in range(num_samples):
                    all_z_vals = []
                    # I added [0] here since the original dim was (1,6379,8)
                    for w in range(1, val_batch.shape[1]+1):
                        z_i_name = z_name + str(w)
                        z_i_val = pred_dict_val[z_i_name][j]
                        all_z_vals.append(z_i_val)
                    all_z_vals = torch.stack(all_z_vals).transpose(0,1)
                    purchase_probs = dmm.predicter(all_z_vals, val_batch_cumtime, val_batch_mask)
                    list_purchase_probs.append(purchase_probs.detach().cpu().numpy())

                purchase_probs = np.mean(np.array(list_purchase_probs), axis=0)
                all_purchase_probs_val = all_purchase_probs_val + list(purchase_probs)


                all_y_val = all_y_val + list(y_val_batch.detach().cpu().numpy())

                if verbose:
                    print("In Validation split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

            if verbose:
                print("Validation Done!                   ")

            roc_auc_val = roc_auc_score(np.array(all_y_val), np.array(all_purchase_probs_val))
        else:
            roc_auc_val = -1

        #MODIFY Test DATA TO CROP LATEST t TIME STEPS
        custom_test_data_sequences = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , test_data_sequences)))
        custom_test_deltatime = np.array(list(map(lambda x: x[:-cropped_t_from_last] , test_deltatime)))
        custom_test_cumtime = np.array(list(map(lambda x: x[:-cropped_t_from_last] , test_cumtime)))

        custom_test_seq_lengths = test_seq_lengths - cropped_t_from_last

        if test_data_sequences_mask is not None:
            custom_test_data_sequences_mask = np.array(list(map(lambda x: x[:-cropped_t_from_last,:] , test_data_sequences_mask)))
        else:
            custom_test_data_sequences_mask = None

        batches_test = batchify(custom_test_data_sequences, custom_test_seq_lengths, custom_test_data_sequences_mask, test_static, y_test, y_test_mask, custom_test_deltatime, custom_test_cumtime, max_len=360 - cropped_t_from_last, batch_size=mini_batch_size, use_feature_mask=use_feature_mask_mini_batch, cuda=args.cuda)

        N_test_data = mini_batch_size * (len(batches_test)-1) + len(batches_test[-1][0])
        N_test_time_slices = float(np.sum(custom_test_seq_lengths))
        eval_N_mini_batches = len(batches_test)


        all_y_test = []
        all_purchase_probs_test = []
        all_purchase_probs_test_std = []
        all_test_batch_indices = [] 
        for i in range(eval_N_mini_batches):
            test_batch_static, test_batch, test_batch_reversed, test_batch_mask, test_batch_seq_lengths, test_batch_feature_mask, y_test_batch, y_mask_test_batch, test_batch_deltatime, test_batch_cumtime, index_test_batch = batches_test[i]

            all_test_batch_indices = all_test_batch_indices + list(index_test_batch)

            pred_dict_test = pred(test_batch_static, test_batch, test_batch_reversed, test_batch_mask,
                                        test_batch_seq_lengths, test_batch_feature_mask, None, None, test_batch_deltatime, test_batch_cumtime, annealing_factor, regularizer, renormalize_y_weight)
            
            #Do Accuracy and ROCAUC analysis for mortality prediction
            list_purchase_probs = []
            z_name = 'z_'
            for j in range(num_samples):
                all_z_tests = []
                for w in range(1, test_batch.shape[1]+1):
                    z_i_name = z_name + str(w)
                    z_i_test = pred_dict_test[z_i_name][j]
                    all_z_tests.append(z_i_test)
                all_z_tests = torch.stack(all_z_tests).transpose(0,1)
                purchase_probs = dmm.predicter(all_z_tests, test_batch_cumtime, test_batch_mask)
                list_purchase_probs.append(purchase_probs.detach().cpu().numpy())

            purchase_probs = np.mean(np.array(list_purchase_probs), axis=0)
            purchase_probs_std = np.std(np.array(list_purchase_probs), axis=0)

            all_purchase_probs_test = all_purchase_probs_test + list(purchase_probs)
            all_purchase_probs_test_std = all_purchase_probs_test_std + list(purchase_probs_std)

            all_y_test = all_y_test + list(y_test_batch.detach().cpu().numpy())
            
            if verbose:
                print("In Test split: %04d/%04d" % (i+1, eval_N_mini_batches), end="\r", flush=True)

        if verbose:
            print("Test Done!                    ")

        roc_auc_test = roc_auc_score(np.array(all_y_test), np.array(all_purchase_probs_test))

        dmm.rnn.train()

        return roc_auc_val, roc_auc_test, np.array(all_y_test), np.array(all_purchase_probs_test), np.array(all_purchase_probs_test_std), np.array(all_test_batch_indices)





    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()
        
    def save_test_results(epoch = -1):
        purchase_predictions_df = pd.DataFrame(columns=["ID_test", "y", "cropped_page", "y_prob", "y_prob_std"])  
        df_save_path = os.path.join(args.experiments_main_folder, args.experiment_folder, "purchase_predictions_test.csv")

        for cropped_t in range(0,50):
            try: 
                if cropped_t == 0:
                    roc_auc_val, roc_auc_test, y_test_np, y_prob_np, y_prob_std_np, index_test = do_evaluation_rocauc(mini_batch_size=args.mini_batch_size, num_samples=10, verbose=False)
                else:
                    roc_auc_val, roc_auc_test, y_test_np, y_prob_np, y_prob_std_np, index_test = do_evaluation_rocauc_custom_time(mini_batch_size=args.mini_batch_size, num_samples=10, cropped_t_from_last=cropped_t, do_eval_for_val=False, verbose=False)

                new_df = pd.DataFrame({"ID_test":index_test, "y":y_test_np, "cropped_page":cropped_t, "y_prob":y_prob_np, "y_prob_std":y_prob_std_np})
                purchase_predictions_df = purchase_predictions_df.append(new_df)

                #Save the latest data frame
                purchase_predictions_df.to_csv(df_save_path, index=False)
                
                if cropped_t == 0:
                    log("ROCAUC [val/test epoch %04d]  %.4f  %.4f" % (epoch, roc_auc_val, roc_auc_test))

                #log("For cropped_t=%d Number of Test Samples: %d" % (cropped_t, len(y_test_np)))
                #log("For cropped_t=%d Test ROC AUC score is : %.4f " % (cropped_t, roc_auc_test))
            except:
                break

    times = [time.time()]
    training_done = False
    #################
    # TRAINING LOOP #
    #################
    if not args.eval_mode: 
        best_val_nll = np.inf
        best_test_nll = np.inf
        val_nll = np.inf
        test_nll = np.inf
        for epoch in range(args.num_epochs):
            # if specified, save model and optimizer states to disk every checkpoint_freq epochs
            if args.save_model != '':
                if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
                    save_checkpoint()
                    #save_checkpoint_trial()

            # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
            epoch_nll = 0.0
            # process each mini-batch; this is where we take gradient steps
            for which_mini_batch in range(N_mini_batches):
                epoch_nll += process_minibatch(epoch, which_mini_batch)

            # report training diagnostics
            times.append(time.time())
            epoch_time = times[-1] - times[-2]
            
            log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
                    (epoch, epoch_nll / float(N_train_data), epoch_time))
            # do evaluation on test and validation data and report results
            if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
                val_nll, test_nll = do_evaluation()
                log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))
                if val_nll < best_val_nll:
                    save_checkpoint(is_best=True)
                    best_val_nll = val_nll
                    
                    #Save the results of the best epoch
                    save_test_results(epoch)
                    
                else:
                    roc_auc_val, roc_auc_test, _, _, _, _ = do_evaluation_rocauc()
                    log("ROCAUC [val/test epoch %04d]  %.4f  %.4f" % (epoch, roc_auc_val, roc_auc_test))
        #training_done = True

    #################
    ### EVALUATION ##
    #################
    if args.eval_mode or training_done:
        if training_done:
            args.load_model = "model_best"
            args.load_opt = "opt_best"
            load_checkpoint()

        purchase_predictions_df = pd.DataFrame(columns=["ID_test", "y", "cropped_page", "y_prob", "y_prob_std"])  
        df_save_path = os.path.join(args.experiments_main_folder, args.experiment_folder, "purchase_predictions_test.csv")

        for cropped_t in range(0,50):
            if cropped_t == 0:
                roc_auc_val, roc_auc_test, y_test_np, y_prob_np, y_prob_std_np, index_test = do_evaluation_rocauc(mini_batch_size=args.mini_batch_size, num_samples=10, verbose=True)
            else:
                roc_auc_val, roc_auc_test, y_test_np, y_prob_np, y_prob_std_np, index_test = do_evaluation_rocauc_custom_time(mini_batch_size=args.mini_batch_size, num_samples=10, cropped_t_from_last=cropped_t, do_eval_for_val=False, verbose=True)

            new_df = pd.DataFrame({"ID_test":index_test, "y":y_test_np, "cropped_page":cropped_t, "y_prob":y_prob_np, "y_prob_std":y_prob_std_np})
            purchase_predictions_df = purchase_predictions_df.append(new_df)

            #Save the latest data frame
            purchase_predictions_df.to_csv(df_save_path, index=False)

            log("For cropped_t=%d Number of Test Samples: %d" % (cropped_t, len(y_test_np)))
            log("For cropped_t=%d Test ROC AUC score is : %.4f " % (cropped_t, roc_auc_test))
            for c in np.arange(0.1,1,0.1):
                pred_label = np.zeros(len(y_prob_np))
                pred_label[y_prob_np>c] = 1

                acc = accuracy_score(y_test_np, pred_label)

                log("Test Accuracy for threshold %.2f : %.4f " % (c,acc))



# parse command-line arguments and execute the main method
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.0')
    torch.set_default_tensor_type('torch.DoubleTensor')

    parser = argparse.ArgumentParser(description="parse args")
    #DMM settings
    parser.add_argument('-zd', '--z_dim', type=int, default=8)
    parser.add_argument('-ed', '--emission_dim', type=int, default=16)
    parser.add_argument('-td', '--transition_dim', type=int, default=32)
    parser.add_argument('-ad', '--att_dim', type=int, default=48)
    parser.add_argument('-md', '--MLP_dims', type=str, default='12-3')
    parser.add_argument('-t2v', '--time2vec_out', type=int, default=8)

    parser.add_argument('-r', '--regularizer', type=float, default=1.0)
    parser.add_argument('--use_feature_mask', action='store_true')
    #Below is only used for masking during ELBO computation (doesn't use mask for emitter or guide!)
    parser.add_argument('--use_feature_mask_ELBO', action='store_true')
    parser.add_argument('--convert_imputations', action='store_true')
    #Label masking stands for semi-supervised experiments
    parser.add_argument('-lm', '--label_masking', type=str, default='')
    #An option to filter out training points that are masked out by label (i.e. label_masking)
    # filter_no_label -> removes data points having no mortality label
    # filter_label -> removes datap points having the mortality label
    # Usage Note: DON'T ACTIVATE both filter_no_label and filter_label
    parser.add_argument('--filter_no_label', action='store_true')
    parser.add_argument('--filter_label', action='store_true')
    #An option to renormalize the weights of y_labels to cover label_masking
    parser.add_argument('--renormalize_y_weight', action='store_true')
    #Add some minimum constant scale to ensure pdf of x will be upper-bounded
    parser.add_argument('-mxs', '--min_x_scale', type=float, default=0.2)
    parser.add_argument('-co', '--clip_obs_val', type=float, default=3.0)
    #Default version is RNN guide. Use below argumment to use GRU instead
    parser.add_argument('--guide_GRU', action='store_true')
    #Linear gain is to predict 'y' at each time step t with linear weight t/T
    #Overall y weights are renormalied to keep balance between reconstruction and prediction loss
    #Renormalization weight is -> (t/T) / (T*(T+1)/(2T))
    parser.add_argument('--linear_gain', action='store_true')
    #Flag for weighted loss
    parser.add_argument('--weighted_loss', action='store_true')

    parser.add_argument('-n', '--num_epochs', type=int, default=200)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip_norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr_decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight_decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini_batch_size', type=int, default=16)
    parser.add_argument('-ae', '--annealing_epochs', type=int, default=100)
    parser.add_argument('-maf', '--minimum_annealing_factor', type=float, default=0.2)
    parser.add_argument('-maxaf', '--maximum_annealing_factor', type=float, default=1.0)
    parser.add_argument('-rdr', '--rnn_dropout_rate', type=float, default=0.0)
    parser.add_argument('-rnl', '--rnn_num_layers', type=int, default=1)
    parser.add_argument('-rd', '--rnn_dim', type=int, default=16)
    parser.add_argument('-iafs', '--num_iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf_dim', type=int, default=100)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=20)
    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('-lopt', '--load_opt', type=str, default='')
    parser.add_argument('-lmod', '--load_model', type=str, default='')
    parser.add_argument('-sopt', '--save_opt', type=str, default='opt')
    parser.add_argument('-smod', '--save_model', type=str, default='model')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    parser.add_argument('-efreq', '--eval_freq', type=int, default=20)
    parser.add_argument('--data_folder', type=str, default='./data/splits0')
    parser.add_argument('--eval_mode', action='store_true')
    parser.add_argument('-nse', '--num_samples_eval', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--tmc', action='store_true')
    parser.add_argument('--tmcelbo', action='store_true')
    parser.add_argument('--tmc_num_samples', default=10, type=int)
    
    args = parser.parse_args()

    if not exists(args.experiments_main_folder):
        os.mkdir(args.experiments_main_folder)
    if not exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
        os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder))

    main(args)


