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
import bouncing_balls
import matplotlib
matplotlib.use('Agg')

# res = 15
# n_balls = 3
# T = 100
# pdb.set_trace()
# sample = bouncing_balls.bounce_vec(res,n=3,T=100)

def test(filenames):
    for filename in filenames:
        print 'Testing parameters: ',filename
        load_dir = base_dir + filename
        split_filename = filename.split('_')
        n_hidden = int(split_filename[0].split('-')[-1])
        n_recurrent = int(split_filename[1].split('-')[-1])
        n_visible = 49
        n_components = int(split_filename[2].split('-')[-1])
        hidden_act = 'sigmoid'

        model = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components,hidden_act='sigmoid',
                          load=True,load_dir=load_dir)
        error = []
        for i in xrange(num_test_sequences):
            seq = mocap_data.sample_test_seq(batch_size) 
            samples = model.sample_given_sequence(seq,num_samples)
            sq_diff = (samples - seq)**2
            sq_diff = sq_diff.mean(axis=0)
            sq_diff = sq_diff.sum(axis=1)
            seq_error = sq_diff.mean(axis=0)
            error.append(seq_error)
            print 'Seq %d error: '%(i+1),seq_error    
        print 'Mean error: ',numpy.mean(error)
        new_results[filename] = error    


results = cPickle.load(open('good_results.pickle','r'))
test_specs = results.keys()
base_dir = '/homes/sss31/PhD/alex_sid_nips/results/'


num_test_sequences = 100
batch_size = 100
num_samples = 5
example = mocap_data.sample_test_seq(batch_size)
new_results = {}

def control_exp():
    n_hidden = 50
    n_recurrent = 25
    n_components = 5
    n_visible = 49
    hidden_act = 'sigmoid'
    model = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components,hidden_act='sigmoid',
                          load=False,)
    error = []
    for i in xrange(num_test_sequences):
        seq = mocap_data.sample_test_seq(batch_size) 
        samples = model.sample_given_sequence(seq,num_samples)
        sq_diff = (samples - seq)**2
        sq_diff = sq_diff.mean(axis=0)
        sq_diff = sq_diff.sum(axis=1)
        seq_error = sq_diff.mean(axis=0)
        error.append(seq_error)
        print 'Seq %d error: '%(i+1),seq_error    
    print 'Mean error: ',numpy.mean(error)
    new_results[filename] = error    
control_exp()
#test(test_specs)