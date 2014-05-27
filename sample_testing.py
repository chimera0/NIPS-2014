import numpy
import theano
from RNADE import *
from RNN_RNADE import RNN_RNADE
import mocap_data
import pdb
from numpy_implementations import *

#model params
n_hidden = 50
n_recurrent = 100
n_visible = 49
n_components = 10
hidden_act = 'sigmoid'

model = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components,hidden_act='sigmoid',
                  load=False,rec_mu=True,rec_mix=False,rec_sigma=False)

load_dir = '/scratch/Sid/RNN_RNADE_2/100-300-10-0.1-0.001/mu'
model.load_model(load_dir,'best_params_train.pickle')
model.build_RNN_RNADE()

batch_size = 100
num_samples = 1
seq = mocap_data.sample_train_seq(batch_size)
new_seq = numpy.zeros(seq.shape)
new_seq[:] = seq
#new_seq = list(new_seq)

ll = theano.function([model.v],[model.log_probs,model.neg_ll,model.u_t,model.b_alpha_t,model.b_mu_t,model.b_sigma_t])

out_ll,neg_ll,u_t,b_alpha_t,b_mu_t,b_sigma_t = ll(new_seq)
#Testing fprop through RNADE for certain parameter configs

rnade = RNADE(n_visible,n_hidden,n_components,hidden_act='sigmoid')
#initialise the values. 
rnade.build_fprop()
rnade.W.set_value(model.W.get_value())
rnade.V_alpha.set_value(model.V_alpha.get_value())
rnade.V_mu.set_value(model.V_mu.get_value())
rnade.V_sigma.set_value(model.V_sigma.get_value())
rnade.activation_rescaling.set_value(model.activation_rescaling.get_value())

inp = new_seq[0,numpy.newaxis]
temp = logdensity_np(inp.T,rnade,b_alpha_t[0],b_mu_t[0],b_sigma_t[0])

rnade_func = theano.function([rnade.v],rnade.ps)

rnade_ll = []
numpy_ll = []
count = 0
for b_alpha,b_mu,b_sigma in zip(b_alpha_t,b_mu_t,b_sigma_t):
	pdb.set_trace()
	inp = new_seq[count,numpy.newaxis]
	#set rnade param values
	rnade.b_alpha.set_value(b_alpha)
	rnade.b_mu.set_value(b_mu)
	rnade.b_sigma.set_value(b_sigma)
	rnade_ll.append(rnade_func(inp))
	numpy_ll.append(logdensity_np(inp.T,rnade,b_alpha_t[0],b_mu_t[0],b_sigma_t[0]))
	count+=1

pdb.set_trace()
