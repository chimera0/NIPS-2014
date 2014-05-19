import os
import sys
import numpy
import cPickle
import theano
import theano.tensor as T 
import state
from RNN_RNADE import RNN_RNADE

import pdb
import mocap_data
import matplotlib
matplotlib.use('Agg')

folder_string = 'nh-10_nr-25_nc-1_l2-0.01_lr-0.0001_lrpt-0.001_trial-0'
load_dir = '/homes/sss31/PhD/alex_sid_nips/results/' + folder_string
folder_name_parts = folder_string.split('_')

n_hidden = int(folder_name_parts[0][-2:])
n_recurrent = int(folder_name_parts[1][-2:])
n_visible = 49
n_components = int(folder_name_parts[2].split('-')[-1])
hidden_act = 'sigmoid'

model = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components,hidden_act='sigmoid',
				  load=True,load_dir=load_dir)

batch_size = 100
num_samples = 10
example = mocap_data.sample_test_seq(batch_size)

samples = model.sample_given_sequence(example,5)
pdb.set_trace()