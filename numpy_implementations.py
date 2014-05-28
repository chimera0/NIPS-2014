import theano
import numpy as np
from utils import *
import pdb



def get_cond_distributions(v_t,model):
    def one_step(x,u_tm1):
        if model.rec_mix:
            shape = model.b_alpha.get_value().shape
            b_alpha = model.b_alpha.get_value().flatten() + numpy.dot(u_tm1,model.Wu_balpha.get_value())
            b_alpha = b_alpha.reshape(shape)
        else:
            b_alpha = model.b_alpha.get_value()
        if model.rec_mu:
            shape = model.b_mu.get_value().shape
            b_mu = model.b_mu.get_value().flatten() + numpy.dot(u_tm1,model.Wu_bmu.get_value())
            b_mu = b_mu.reshape(shape)
        else:
            b_mu = model.b_mu.get_value()
        if model.rec_sigma:
            shape = model.b_sigma.get_value().shape
            b_sigma = model.b_sigma.get_value().flatten() + numpy.dot(u_tm1,model.Wu_bsigma.get_value())
            b_sigma = b_sigma.reshape(shape)
        else:
            b_sigma = model.b_sigma.get_value()
        u = numpy.tanh(model.bu.get_value() + numpy.dot(x,model.Wvu.get_value()) + numpy.dot(u_tm1,model.Wuu.get_value()))
        return u,b_alpha,b_mu,b_sigma
    u_t = []
    b_alpha_t = []
    b_mu_t = []
    b_sigma_t = []
    for i in xrange(v_t.shape[0]):
        if i==0:
            u,b_alpha,b_mu,b_sigma = one_step(v_t[i],model.u0.get_value())
            u_t.append(u)
            b_alpha_t.append(b_alpha)
            b_mu_t.append(b_mu)
            b_sigma_t.append(b_sigma)
        else:
            u,b_alpha,b_mu,b_sigma = one_step(v_t[i],u_t[-1])
            u_t.append(u)
            b_alpha_t.append(b_alpha)
            b_mu_t.append(b_mu)
            b_sigma_t.append(b_sigma)
    return numpy.array(u_t),numpy.array(b_alpha_t),numpy.array(b_mu_t),numpy.array(b_sigma_t)



def RNN_RNADE_fprop(v_t,model):
    u_t,b_alpha_t,b_mu_t,b_sigma_t = get_cond_distributions(v_t,model)
    ll = []
    i = 0
    for b_alpha,b_mu,b_sigma in zip(b_alpha_t,b_mu_t,b_sigma_t):
        inp = v_t[i]
        inp = inp[:,numpy.newaxis] 
        log_lik = numpy_rnade(inp,model,b_alpha,b_mu,b_sigma)
        ll.append(log_lik)
        i+=1
    return numpy.array(ll),u_t,b_alpha_t,b_mu_t,b_sigma_t


def numpy_rnade(x,model,b_alpha,b_mu,b_sigma):
    W = model.W.get_value()
    V_alpha = model.V_alpha.get_value()
    V_mu = model.V_mu.get_value()
    V_sigma = model.V_sigma.get_value()
    activation_rescaling = model.activation_rescaling.get_value()
    k = 0.5*np.log(2*np.pi)
    a = 0
    p = 0
    print 'input vector'
    print x
    for i in xrange(x.shape[0]):
        if i == 0:
            a = W[i] 
            tmp = a
            x_prev = 1
        else:
            tmp = W[i] * x[i-1] 
            a = a + tmp
            x_prev = x[i-1] 
        print 'x_prev'
        print x_prev
        print 'w'
        print W[i]
        print 'tmp'
        print tmp
        print 'a:'
        print a
        activations = a * activation_rescaling[i]
        h = sigmoid(activations)
        print 'h:'
        print h
        alpha = softmax(numpy.dot(h,V_alpha[i]) + b_alpha[i])
        print 'alhpa'
        print alpha
        mu = numpy.dot(h,V_mu[i]) + b_mu[i]
        print 'mu'
        print mu
        sigma = numpy.exp(numpy.dot(h,V_sigma[i]) + b_sigma[i])
        print 'sigma'
        print sigma
        arg = -0.5 * (((mu - x[i]) / sigma)**2) + np.log(alpha) + (-np.log(sigma)-k)
        print 'arg'
        print arg
        inc = logsumexp(arg)
        p = p + inc
        print 'p'
        print p
        #print inc
        #print p
    return p