import numpy
import theano
from RNADE import *
from RNN_RNADE import RNN_RNADE
import mocap_data
import pdb

def sigmoid(s):
    s = 1.0 / (1.0 + numpy.exp(-1.0 * s))
    return s

def softmax(X):
    """Calculates softmax row-wise"""
    if X.ndim == 1:
        X = X - numpy.max(X)
        e = numpy.exp(X)
        return e/e.sum()
    else:
        X = X - numpy.max(X,1)[:,numpy.newaxis]
        e = numpy.exp(X)    
        return e/e.sum(1)[:,numpy.newaxis]


n_hidden = 50
n_visible = 49
n_components = 10
hidden_act = 'sigmoid'

rnade = RNADE(n_visible,n_hidden,n_components,hidden_act='sigmoid')
rnade.build_fprop()
rnade_func = theano.function([rnade.v],[rnade.ps,rnade.cost])

inp = numpy.random.random((2,49))
probs,cost = rnade_func(inp)
pdb.set_trace()

# def density_given_previous_a_and_x(x, w, V_alpha, b_alpha, V_mu, b_mu, V_sigma, b_sigma,activation_factor, p_prev, a_prev, x_prev,):
#     a = a_prev + numpy.dot(x_prev[:,numpy.newaxis], w[numpy.newaxis,:])
#     h = sigmoid(a * activation_factor)  # BxH
#     Alpha = softmax(numpy.dot(h, V_alpha) + b_alpha[numpy.newaxis,:])  # BxC
#     Mu = numpy.dot(h, V_mu) + T.shape_padleft(b_mu)  # BxC
#     Sigma = T.exp((T.dot(h, V_sigma) + T.shape_padleft(b_sigma)))  # BxC
#     p = p_prev + log_sum_exp(-constantX(0.5) * T.sqr((Mu - T.shape_padright(x, 1)) / Sigma) - T.log(Sigma) - constantX(0.5 * numpy.log(2 * numpy.pi)) + T.log(Alpha))
#     return (p, a, x)



# def fprop(x,model,batch_size=20):
# 	a0 = numpy.zeros((batch_size,n_hidden))
# 	p0 = numpy.zeros_like(x[0])
# 	x0 = numpy.zeros_like(x[0])




