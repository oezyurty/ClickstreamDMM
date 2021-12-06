# ClickstreamDMM

## Dependencies 

For a straight-forward use of *Clickstream*DMM, you can install the required libraries from *requirements.txt*:
`pip install -r requirements.txt`

## Dataset

We made our experiments on a clickstream dataset provided by our partner company. We cannot release the dataset, however, we suggest having the following files for a straight-forward usage of our algorithm.

To be aligned with our AttDMM *Clickstream*DMM, you need to have the following files under each split directory.
1. Time-series of pages:
	1. timeseries_train.npy
	1. timeseries_val.npy
	1. timeseries_test.npy
1. Time-series of log TSP (time spent on each page):
	1. delta_time_train_log.npy
	1. delta_time_val_log.npy
	1. delta_time_test_log.npy
1. Time-series of cumulative time up to the page:
	1. cum_time_train.npy
	1. cum_time_val.npy
	1. cum_time_test.npy
1. Static features:
	1. static_train.npy
	1. static_val.npy
	1. static_test.npy
1. Purchase Labels:
	1. y_train.npy
	1. y_val.npy
	1. y_test.npy

## Example Usage

For training:
`python main.py --cuda --experiments_main_folder experiments --experiment_folder default --log clickstreamdmm.log --save_model model --save_opt opt --checkpoint_freq 10 --eval_freq 10 --data_folder /home/Data/fold0`

All the log files and the model checkpoints will be saved under *current_dir/experiments_main_folder/experiment_folder/*

for testing:
`python main.py --cuda --experiments_main_folder experiments --experiment_folder default --log clickstreamdmm_eval.log --load_model model_best --load_opt opt_best --eval_mode --data_folder /home/Data/fold0`

Note that *experiments_main_folder* and *experiment_folder* have to be consistent with training so that the correct model is loaded properly.  After testing is done, the prediction outputs can be found as *current_dir/experiments_main_folder/experiment_folder/purchase_predictions_test.csv*

For the full set of arguments, please check main.py .
