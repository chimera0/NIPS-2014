import numpy
import theano
from RNADE import *
from RNN_RNADE import RNN_RNADE
import mocap_data
import pdb

#model params
n_hidden = 100
n_recurrent = 300
n_visible = 49
n_components = 2
hidden_act = 'sigmoid'

model = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components,hidden_act='sigmoid',
                  load=False,)

load_dir = '/scratch/Sid/RNN-RNADE/100-300-2-0.01-0.001/1'
model.load_model(load_dir)


batch_size = 100
num_samples = 1
seq = mocap_data.sample_train_seq(batch_size)
sample = model.sample_given_sequence(seq,1)

diff = (seq - sample)**2
pdb.set_trace()