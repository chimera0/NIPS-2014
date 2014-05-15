"""
Class that builds the symbolic graph for the RNN-RNADE.
Siddharth Sigtia
April, 2014
C4DM
"""
import numpy
import cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from model import Model
from datasets import Dataset
import mocap_data
from SGD import SGD_Optimiser
import pdb
from RNADE import *

def shared_normal(shape, scale=1,name=None):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(name=name,value=numpy.random.normal(
    scale=scale, size=shape).astype(theano.config.floatX))

def shared_zeros(shape,name=None):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(name=name,value=numpy.zeros(shape, dtype=theano.config.floatX))

def sigmoid(x):
  return 0.5*numpy.tanh(0.5*x) + 0.5

def constantX(value):
    """
    Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value, dtype=theano.config.floatX))

def log_sum_exp(x, axis=1):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))

floatX = theano.config.floatX

class RNN_RNADE(Model):
    def __init__(self,n_visible,n_hidden,n_recurrent,n_components,hidden_act='ReLU',
                 l2=1.,rec_sigma=True,rec_mu=True,rec_mix=True,load=False,load_dir=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_recurrent = n_recurrent
        self.n_components = n_components
        #RNADE params
        self.W = shared_normal((n_visible, n_hidden), 0.01,'W')
        self.b_alpha = shared_normal((n_visible,n_components),0.01,'b_alpha')
        self.V_alpha = shared_normal((n_visible,n_hidden,n_components),0.01,'V_alpha')
        self.b_mu = shared_normal((n_visible,n_components),0.01,'b_mu')
        self.V_mu = shared_normal((n_visible,n_hidden,n_components),0.01,'V_mu')
        #From Benigno Uria's implementation
        self.b_sigma = shared_normal((n_visible,n_components),0.01,'b_sigma')
        self.b_sigma.set_value(self.b_sigma.get_value() + 1.0)

        self.V_sigma = shared_normal((n_visible,n_hidden,n_components),0.01,'V_sigma')
        #Initialising activation rescaling to all 1s. 
        self.activation_rescaling = shared_zeros((n_visible,),'activation_rescaling')
        self.activation_rescaling.set_value(self.activation_rescaling.get_value() + 1.0)
        #RNN params
        self.Wuu = shared_normal((n_recurrent, n_recurrent), 0.0001,'Wuu')
        self.Wvu = shared_normal((n_visible, n_recurrent), 0.0001,'Wvu')
        self.bu = shared_zeros((n_recurrent),'bu')
        self.u0 = shared_zeros((n_recurrent),'u0')
        #RNN-RNADE params, not all of them are used.
        self.Wu_balpha = shared_normal((n_recurrent,n_visible*n_components),0.01,'Wu_balpha')
        self.Wu_bmu = shared_normal((n_recurrent,n_visible*n_components),0.01,'Wu_bmu')
        self.Wu_bsigma = shared_normal((n_recurrent,n_visible*n_components),0.01,'Wu_bsigma')
        self.Wu_Valpha = shared_normal((n_recurrent,n_visible*n_hidden*n_components),0.01,'Wu_Valpha')
        self.Wu_Vmu = shared_normal((n_recurrent,n_visible*n_hidden*n_components),0.01,'Wu_Vmu')
        self.Wu_Vsigma = shared_normal((n_recurrent,n_visible*n_hidden*n_components),0.01,'Wu_Vsigma')
        self.params = [self.W,self.b_alpha,self.V_alpha,self.b_mu,self.V_mu,self.b_sigma,self.V_sigma,self.activation_rescaling,self.Wuu,
                       self.bu]#,self.Wu_balpha,self.Wu_bmu,self.Wu_bsigma]
        #params to decide the architecture
        self.rec_sigma = rec_sigma
        self.rec_mu = rec_mu
        self.rec_mix = rec_mix
        if rec_sigma:
            self.params.append(self.Wu_bsigma)
        if rec_mu:
            self.params.append(self.Wu_bmu)
        if rec_mix:
            self.params.append(self.Wu_balpha)

        self.params_dict = {}
        for param in self.params:
            self.params_dict[param.name] = param
        self.l2 = l2
        #input sequence
        self.v = T.matrix('v')
        self.hidden_act = hidden_act
        if self.hidden_act == 'sigmoid':
            self.nonlinearity = T.nnet.sigmoid
        elif self.hidden_act == 'ReLU':
            self.nonlinearity = lambda x:T.maximum(x,0.)
        #Parameters for loading
        self.load = load
        self.load_dir = load_dir
        if self.load:
            self.load_model(self.load_dir)

    def rnade_sym(self,x,W,V_alpha,b_alpha,V_mu,b_mu,V_sigma,b_sigma,activation_rescaling):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        def density_given_previous_a_and_x(x, w, V_alpha, b_alpha, V_mu, b_mu, V_sigma, b_sigma,activation_factor, p_prev, a_prev, x_prev,):
            a = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
            h = self.nonlinearity(a * activation_factor)  # BxH

            Alpha = T.nnet.softmax(T.dot(h, V_alpha) + T.shape_padleft(b_alpha))  # BxC
            Mu = T.dot(h, V_mu) + T.shape_padleft(b_mu)  # BxC
            Sigma = T.exp((T.dot(h, V_sigma) + T.shape_padleft(b_sigma)))  # BxC
            p = p_prev + log_sum_exp(-constantX(0.5) * T.sqr((Mu - T.shape_padright(x, 1)) / Sigma) - T.log(Sigma) - constantX(0.5 * numpy.log(2 * numpy.pi)) + T.log(Alpha))
            return (p, a, x)
        # First element is different (it is predicted from the bias only)
        a0 = T.zeros_like(T.dot(x.T, W))  # BxH
        #a0 = T.cast(a0,floatX)
        p0 = T.zeros_like(x[0])
        #p0 = T.cast(p0,floatX)
        x0 = T.ones_like(x[0])
        #x0 = T.cast(x0,floatX)
        
        ([ps, _as, _xs], updates) = theano.scan(density_given_previous_a_and_x,
                                                sequences=[x, W, V_alpha, b_alpha,V_mu,b_mu,V_sigma,b_sigma,activation_rescaling],
                                                outputs_info=[p0, a0, x0])
        return (ps[-1], updates)

    def get_cond_distributions(self,v_t):
        def one_step(v_t,u_tm1):
            if self.rec_mix:
                b_alpha = self.b_alpha.get_value().flatten() + numpy.dot(u_tm1,self.Wu_balpha.get_value())
            else:
                b_alpha = self.b_alpha.get_value().get_value().flatten()
            if self.rec_mu:
                b_mu = self.b_mu.get_value().flatten() + numpy.dot(u_tm1,self.Wu_bmu.get_value())
            else:
                b_mu = self.b_mu.get_value().flatten()
            if self.rec_sigma:
                b_sigma = self.b_sigma.get_value().flatten() + numpy.dot(u_tm1,self.Wu_bsigma.get_value())
            else:
                b_sigma = self.b_sigma.get_value().flatten()
            u = numpy.tanh(self.bu.get_value() + numpy.dot(v_t,self.Wvu.get_value())) + numpy.dot(u_tm1,self.Wuu.get_value())
            return u,b_alpha,b_mu,b_sigma
        u_t = []
        b_alpha_t = []
        b_mu_t = []
        b_sigma_t = []
        for i in xrange(v_t.shape[0]):
            if i==0:
                u,b_alpha,b_mu,b_sigma = one_step(v_t[i],self.u0.get_value())
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

    
    def recurrence(self,x,u_tm1):
        #Flattening the array so that dot product is easier. 
        if self.rec_mix:
            b_alpha_t = self.b_alpha.flatten(ndim=1) + T.dot(u_tm1,self.Wu_balpha)
        else:
            b_alpha_t = self.b_alpha.flatten(ndim=1)
        if self.rec_mu:
            b_mu_t = self.b_mu.flatten(ndim=1) + T.dot(u_tm1,self.Wu_bmu)
        else:
            b_mu_t = self.b_mu.flatten(ndim=1)
        if self.rec_sigma:
            b_sigma_t = self.b_sigma.flatten(ndim=1) + T.dot(u_tm1,self.Wu_bsigma)
        else:
            b_sigma_t = self.b_sigma.flatten(ndim=1)
        u_t = T.tanh(self.bu + T.dot(x,self.Wvu)) + T.dot(u_tm1,self.Wuu)
        #Reshape all the time dependent arrays
        b_alpha_t = b_alpha_t.reshape(self.b_alpha.shape)
        b_mu_t = b_mu_t.reshape(self.b_mu.shape)
        b_sigma_t = b_sigma_t.reshape(self.b_sigma.shape)
        inp = T.shape_padright(x) #Padding the input, it should be (V X 1)
        prob,updates = self.rnade_sym(inp,self.W,self.V_alpha,b_alpha_t,self.V_mu,b_mu_t,self.V_sigma,b_sigma_t,self.activation_rescaling)
        return u_t,b_alpha_t,b_mu_t,b_sigma_t,prob

    def build_RNN_RNADE(self,):
        #(u_t,b_alpha_t,b_mu_t,b_sigma_t),updates = theano.scan(self.rnn_recurrence,sequences=self.v,outputs_info=[self.u0,None,None,None])
        #self.log_probs,updates = theano.scan(self.rnade_recurrence,sequences=[self.v,b_alpha_t,b_mu_t,b_sigma_t],outputs_info=[None])
        print 'Building computational graph for the RNN_RNADE.'
        (u_t,b_alpha_t,b_mu_t,b_sigma_t,self.log_probs),updates = theano.scan(self.recurrence,sequences=self.v,outputs_info=[self.u0,None,None,None,None])
        self.neg_ll = -self.log_probs
        self.neg_ll_cost = T.mean(self.neg_ll,axis=0)
        self.cost = T.mean(self.neg_ll) + self.l2*T.sum(self.W**2) #Average negative log-likelihood per frame
        self.l2_cost = T.sum(self.W**2)
        print 'Done building graph.'
    
    def sample_RNADE(self,b_alpha,b_mu,b_sigma,n):
        W = self.W.get_value()
        V_alpha = self.V_alpha.get_value()
        #b_alpha = b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        #b_mu = b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        #b_sigma = b_sigma.get_value()
        activation_rescaling = self.activation_rescaling.get_value()
        samples = numpy.zeros((self.n_visible, n))
        for s in xrange(n):
            a = numpy.zeros((self.n_hidden,))  # H
            for i in xrange(self.n_visible):
                if i == 0:
                    a = W[i, :]
                else:
                    a = a + W[i, :] * samples[i - 1, s]
                h = activation_function[self.hidden_act](a * activation_rescaling[i])
                alpha = softmax(numpy.dot(h, V_alpha[i]) + b_alpha[i])  # C
                Mu = numpy.dot(h, V_mu[i]) + b_mu[i]  # C
                Sigma = numpy.minimum(numpy.exp(numpy.dot(h, V_sigma[i]) + b_sigma[i]), 1)
                comp = random_component(alpha)
                samples[i, s] = numpy.random.normal(Mu[comp], Sigma[comp])
        return samples.T

    def test(self,):
        pdb.set_trace()
        self.test_func = theano.function([self.v],self.log_probs)
        test_seq = numpy.random.random((100,49)) 
        temp = self.test_func(test_seq)
        #(u_t,b_alpha_t,b_mu_t,b_sigma_t),updates = theano.scan(self.rnn_recurrence,sequences=self.v,outputs_info=[self.u0,None,None,None])
        #a,b,c,d = self.test_func(test_seq)
        #e,f,g,h = self.get_cond_distributions(test_seq)
        
    def sample_given_sequence(self,seq,n):
        u_t,b_alpha_t,b_mu_t,b_sigma_t = self.get_cond_distributions(seq)
        all_sequences = []
        for i in xrange(n):
            sequence = []
            for b_alpha,b_mu,b_sigma in zip(b_alpha_t,b_mu_t,b_sigma_t):
                sequence.append(self.sample_RNADE(b_alpha,b_mu,b_sigma,1))
            sequence = numpy.array(sequence).reshape((-1,self.n_visible))
            
            all_sequences.append(sequence)
        return numpy.array(all_sequences)

if __name__ == '__main__':
    n_visible = 49
    n_hidden = 50
    n_recurrent = 30
    n_components = 2
    test = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components)
    test.build_RNN_RNADE()
    test.test()
    #test.init_RNADE()
    #test_sequence = numpy.random.random((100,49))
    #test_func = theano.function([test.v],test.probs)
    #input_sequence = []
    #for i in xrange(100):
    #    input_sequence.append(numpy.random.random(10))

    #probs= test.test_func(input_sequence)
    #pdb.set_trace()
    