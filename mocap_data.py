import numpy
import scipy.io 
import pdb
#reading data from .mat file
A = scipy.io.loadmat('data/MOCAP')
data = A['batchdata']
seqlengths = A['seqlengths'][0]

seqstarts = numpy.concatenate(([0], numpy.cumsum(seqlengths)))
seqprobs = seqlengths / float(seqlengths.sum())

batch_size = 100
test_frac = 0.2

def int_rand_range(a,b):
	return int(numpy.floor(numpy.random.rand()*(b-a) + a))

def sample_seq(batch_size):
	seq_id = numpy.random.multinomial(1,seqprobs).argmax()
	seq_len = seqlengths[seq_id]
	seq_start = seqstarts[seq_id]
	seq_end = seqstarts[seq_id+1]
	rand_pos = int_rand_range(seq_start,seq_end-batch_size)
	return data[rand_pos:rand_pos+batch_size]


def sample_train_seq(batch_size):
	seq_id = numpy.random.multinomial(1,seqprobs).argmax()
	seq_len = seqlengths[seq_id]
	seq_start = seqstarts[seq_id]
	seq_end = seqstarts[seq_id+1]
	assert(seq_len == seq_end - seq_start)
	seq_end = seq_end - seq_len*test_frac
	rand_pos = int_rand_range(seq_start,seq_end-batch_size)
	return data[rand_pos:rand_pos+batch_size]

def sample_test_seq(batch_size):
	seq_id = numpy.random.multinomial(1,seqprobs).argmax()
	seq_len = seqlengths[seq_id]
	seq_start = seqstarts[seq_id]
	seq_end = seqstarts[seq_id+1]
	assert(seq_len == seq_end - seq_start)
	seq_start = seq_end - seq_len*test_frac
	rand_pos = int_rand_range(seq_start,seq_end-batch_size)
	return data[rand_pos:rand_pos+batch_size]

rand_seq = sample_seq(50)
train_seq = sample_train_seq(50)
test_seq = sample_test_seq(50)
