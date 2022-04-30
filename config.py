#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py
we are using the same experiment configuration from
author: xhchrn
        chernxh@tamu.edu
Set up experiment configuration using argparse library.
"""

import os
import sys
import datetime
import argparse
from math import ceil

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()

# Network arguments
net_arg = parser.add_argument_group('net')
net_arg.add_argument(
    '-n', '--net', type=str,
    help='Network name.')
net_arg.add_argument(
    '-T', '--T', type=int, default=16,
    help="Number of layers of iterative algorithms.")
net_arg.add_argument(
    '-l', '--lam', type=float, default=0.4,
    help="Initial lambda in LISTA solvers.")
net_arg.add_argument(
    '-nums', '--numbers', type=int, default=16,
    help="Initial numbers of shared LISTA's weights.")
net_arg.add_argument(
    '-is_us', '--is_unshared', type=bool, default=False,
    help="Whether weights are unshared between layers.")
net_arg.add_argument(
    '-is_na', '--is_na', type=bool, default=False,
    help="If add Nesterov Acceleration.")
net_arg.add_argument(
    '-has_ss', '--has_ss', type=bool, default=False,
    help="If support support selection.")
net_arg.add_argument(
    '-p', '--percent', type=float, default=0.8,
    help="Percent of entries to be selected as support in each layer.")
net_arg.add_argument(
    '-maxp', '--max_percent', type=float, default=0.0,
    help="Maximum percentage of entries to be selected as support in each layer.")


#### for GLISTA
net_arg.add_argument(
    '-uf', '--uf', type=str, default='combine',
    help='Task type, in [`combine`, `inv`].')
net_arg.add_argument(
    '-a', '--alti', type=float, default=1.0,
    help="")
net_arg.add_argument(
    '-ov', '--overshoot', type=bool, default=True,
    help="Whether using overshoot strategy")
net_arg.add_argument(
    '-ga', '--gain', type=bool, default=True,
    help="Whether using gain strategy")
net_arg.add_argument(
    '-gf', '--gain_fun', type=str, default='combine',
    help='.')
net_arg.add_argument(
    '-of', '--over_fun', type=str, default='none',
    help='.')
net_arg.add_argument(
    '-bg', '--both_gate', type=bool, default=True,
    help="Using two gate strategies")
net_arg.add_argument(
    '-Tc', '--T_combine', type=int, default=9,
    help="")
net_arg.add_argument(
    '-Tm', '--T_middle', type=int, default=30,
    help="")


# Experiments arguments
exp_arg = parser.add_argument_group('exp')
exp_arg.add_argument(
    '-id', '--exp_id', type=int, default=0,
    help="ID of the experiment/model.")
exp_arg.add_argument(
    '-ef', '--exp_folder', type=str, default='./experiments',
    help="Folder for saving model parameters.")
exp_arg.add_argument(
    '-pf', '--prob_folder', type=str, default='',
    help="Subfolder in exp_folder for a specific setting of problem.")
exp_arg.add_argument(
    '-rf', '--res_folder', type=str, default='./results',
    help="Experiments' results folder for saving model results.")
exp_arg.add_argument(
    '-df', '--data_folder', type=str, default='./data',
    help="Root folder where trix A and test data are saved.")

exp_arg.add_argument(
    '-t', '--test', action='store_true',
    help="Flag of training or testing models.")

exp_arg.add_argument(
    '-is_cn', '--is_col_normalized', type=bool, default=True,
    help="Whether each column of Matrix A need normalization.")

####### log
exp_arg.add_argument(
    '-log', '--log', type=bool, default='True',
    help="Use tensorboard to record training loss.")
exp_arg.add_argument(
    '-lf', '--log_folder', type=str, default='./tensorboard',
    help="Root folder where trix A and test data are saved.")


# Problem arguments
prob_arg = parser.add_argument_group('prob')
prob_arg.add_argument(
    '-task', '--task_type', type=str, default='sc',
    help='Task type, in [`gd`, `str`, `ste`, `sc`, `cs`, `denoise`]. '
         'gd: generate data, "str": simulated training, "ste": simulated test,'
         '')
prob_arg.add_argument(
    '-M', '--M', type=int, default=250,
    help="Dimension of measurements.")
prob_arg.add_argument(
    '-N', '--N', type=int, default=500,
    help="Dimension of sparse codes.")
prob_arg.add_argument(
    '-has_No', '--has_noise', type=bool, default=False,
    help="Whether noise is added.")
prob_arg.add_argument(
    '-S', '--SNR', type=int, default=40,
    help="Strength of noises in measurements.")


"""Training arguments."""
train_arg = parser.add_argument_group('train')
train_arg.add_argument(
    '-ld', '--lr_decay', type=str, default='1,0.2,0.02',
    help="Learning rate decaying rate after training each layer.")
train_arg.add_argument(
    '-vbs', '--vbs', type=int, default=1000,
    help="Validation batch size.")
train_arg.add_argument(
    '-supp', '--support', type=float, default=0.1,
    help="The proportion supports of sparse solvers.")
train_arg.add_argument(
    '-mo', '--model', type=str, default="",
    help="File name according to different model.")
train_arg.add_argument(
    '-lr', '--init_lr', type=float, default=5e-4,
    help="Initial learning rate.")
train_arg.add_argument(
    '-ep', '--epochs', type=int, default=200000,
    help="Epochs.")
train_arg.add_argument(
    '-tbs', '--tbs', type=int, default=64,
    help="Training batch size.")
train_arg.add_argument(
    '-bw', '--better_wait', type=int, default=4000,
    help="Waiting time before jumping to next stage.")



# train_arg.add_argument(
#     "-epoch", "--num_epochs", type=int, default=20,
#     help="The number of training epochs for denoise experiments.\n"
#          "`-1` means infinite number of epochs.")



################################
###### face recognition  #######
################################
net_arg.add_argument(
    '-is_ed', '--is_en_de', type=bool, default=False,
    help="if encoder and decoder")
net_arg.add_argument(
    '-dc2n', '--D_cl2num', type=str, default="",
    help="dict of classes to nums")
net_arg.add_argument(
    '-ct', '--classify_type', type=str, default="SRC",
    help="get feature for face recognition in [SRC, dense, Cosine_simi, CSC, double_LISTA_net]")
net_arg.add_argument(
    '-cl', '--classes', type=int, default=38,
    help="number of classes")
net_arg.add_argument(
    '-dt', '--data_type', type=str, default="Yale",
    help="get databases for face recognition in [Yale, AR]")
net_arg.add_argument(
    '-ft', '--feature_type', type=str, default="random",
    help="get feature for face recognition in [random, downsampling]")
net_arg.add_argument(
    '-fs', '--feature_shape', type=int, default=120,
    help="get feature shape for face recognition in [30, 56, 120, 504]")


train_arg.add_argument(
    '-tr_sam', '--tr_samples', type=int, default=64,
    help="Initial total samples.")
train_arg.add_argument(
    '-val_sam', '--val_samples', type=int, default=1000,
    help="Initial total samples.")


def get_config():
    config, unparsed = parser.parse_known_args()

    """
    Check validity of arguments.
    """
    # check if a network model is specified
    if config.task_type != "gd":
        if config.net is None:
            raise ValueError('no model specified')

    # set experiment path and folder
    if not os.path.exists(config.exp_folder):
        os.mkdir(config.exp_folder)

    # make experiment base path and results base path
    setattr(config, 'exp_base', os.path.join(config.exp_folder,
                                            config.prob_folder))
    setattr(config, 'res_base', os.path.join(config.res_folder,
                                            config.prob_folder))
    setattr(config, 'data_base', os.path.join(config.data_folder,
                                            config.prob_folder))
    if not os.path.exists(config.exp_base):
        os.mkdir(config.exp_base)
    if not os.path.exists(config.res_base):
        os.mkdir(config.res_base)
    if not os.path.exists(config.data_base):
        os.mkdir(config.data_base)

    # lr_decay
    config.lr_decay = tuple([float(decay) for decay in config.lr_decay.split(',')])

    return config, unparsed

