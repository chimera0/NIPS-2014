import numpy
import theano
from RNADE import *
from RNN_RNADE import RNN_RNADE
import mocap_data
import pdb
from numpy_implementations import *
import pickle

batch_size = 100
num_samples = 1
seq = mocap_data.sample_train_seq(batch_size)
new_seq = numpy.zeros(seq.shape)
new_seq[:] = seq
pickle.dump(new_seq,open('seq.pkl','w'))
