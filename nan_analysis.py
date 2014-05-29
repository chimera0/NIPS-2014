import os
import numpy
import theano
from RNADE import *
from RNN_RNADE import RNN_RNADE
import mocap_data
import pdb
from numpy_implementations import *
import pickle

#model params
n_hidden = 50
n_recurrent = 100
n_visible = 49
n_components = 10
hidden_act = 'sigmoid'

model = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components,hidden_act='sigmoid',
                  load=False,rec_mu=True,rec_mix=False,rec_sigma=False)

load_dir = '/scratch/Sid/RNN_RNADE_fast_reg/100-300-1-0.001-0.1/mu-sigma'
model.load_model(load_dir,'params_NaN.pickle')
model.build_RNN_RNADE()

# batch_size = 100
# num_samples = 1
# seq = mocap_data.sample_train_seq(batch_size)
# new_seq = numpy.zeros(seq.shape)
# new_seq[:] = seq
path = os.path.join(load_dir,'nan_seq.pkl')
new_seq = pickle.load(open(path,'r'))

ll = RNN_RNADE_fprop(new_seq,model)
pdb.set_trace()