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
import pdb

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
    def __init__(self,n_visible,n_hidden,n_recurrent,n_components,hidden_act='ReLU',):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_recurrent = n_recurrent
        #RNADE params
        self.W = shared_normal((n_visible, n_hidden), 0.01,'W')
        self.b_alpha = shared_normal((n_visible,n_components),0.01,'b_alpha')
        self.V_alpha = shared_normal((n_visible,n_hidden,n_components),0.01,'V_alpha')
        self.b_mu = shared_normal((n_visible,n_components),0.01,'b_mu')
        self.V_mu = shared_normal((n_visible,n_hidden,n_components),0.01,'V_mu')
        self.b_sigma = shared_normal((n_visible,n_components),0.01,'b_sigma')
        self.V_sigma = shared_normal((n_visible,n_hidden,n_components),0.01,'V_sigma')
        self.activation_rescaling = shared_normal((n_visible),0.01,'activation_rescaling')
        #RNN params
        self.Wuu = shared_normal((n_recurrent, n_recurrent), 0.0001,'Wuu')
        self.Wvu = shared_normal((n_visible, n_recurrent), 0.0001,'Wvu')
        self.bu = shared_zeros((n_recurrent),'bu')
        self.u0 = shared_zeros((n_recurrent),'u0')
        #RNN-RNADE params
        self.Wu_balpha = shared_normal((n_recurrent,n_visible*n_components),0.01,'Wu_balpha')
        self.Wu_bmu = shared_normal((n_recurrent,n_visible*n_components),0.01,'Wu_bmu')
        self.Wu_bsigma = shared_normal((n_recurrent,n_visible*n_components),0.01,'Wu_bsigma')
        self.Wu_Valpha = shared_normal((n_recurrent,n_visible*n_hidden*n_components),0.01,'Wu_Valpha')
        self.Wu_Vmu = shared_normal((n_recurrent,n_visible*n_hidden*n_components),0.01,'Wu_Vmu')
        self.Wu_Vsigma = shared_normal((n_recurrent,n_visible*n_hidden*n_components),0.01,'Wu_Vsigma')
        self.params = [self.W,self.b_alpha,self.V_alpha,self.b_mu,self.V_mu,self.b_sigma,self.V_sigma,self.activation_rescaling,self.Wuu,
                       self.bu]
        #input sequence
        self.v = T.matrix('v')
        self.hidden_act = hidden_act
        if self.hidden_act == 'sigmoid':
            self.nonlinearity = T.nnet.sigmoid
        elif self.hidden_act == 'ReLU':
            self.nonlinearity = lambda x:T.maximum(x,0.)

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

    def recurrence(self,v_t,u_tm1):
        #Flattening the array so that dot product is easier. 
        b_alpha_t = self.b_alpha.flatten(ndim=1) + T.dot(u_tm1,self.Wu_balpha)
        b_mu_t = self.b_mu.flatten(ndim=1) + T.dot(u_tm1,self.Wu_bmu)
        b_sigma_t = self.b_sigma.flatten(ndim=1) + T.dot(u_tm1,self.Wu_bsigma)
        u_t = T.tanh(self.bu + T.dot(v_t,self.Wvu)) + T.dot(u_tm1,self.Wuu)
        return u_t,b_alpha_t,b_mu_t,b_sigma_t

    def rnade_recurrence(self,x,b_alpha_t,b_mu_t,b_sigma_t):
        #Reshape all the time dependent arrays
        b_alpha_t = b_alpha_t.reshape(self.b_alpha.shape)
        b_mu_t = b_mu_t.reshape(self.b_mu.shape)
        b_sigma_t = b_sigma_t.reshape(self.b_sigma.shape)
        inp = T.shape_padright(x) #Padding the input, it should be (V X 1), I think by default the shape afer scan is (V,) which causes 
                                  #issues with broadcasting and accumulating the results. 
        prob,updates = self.rnade_sym(inp,self.W,self.V_alpha,b_alpha_t,self.V_mu,b_mu_t,self.V_sigma,b_sigma_t,self.activation_rescaling)
        return prob
    
    def build_RNN_RNADE(self,):
        (u_t,b_alpha_t,b_mu_t,b_sigma_t),updates = theano.scan(self.recurrence,sequences=self.v,outputs_info=[self.u0,None,None,None])
        self.probs,updates = theano.scan(self.rnade_recurrence,sequences=[self.v,b_alpha_t,b_mu_t,b_sigma_t],outputs_info=[None])
        self.cost = T.mean(self.probs)#self.probs.sum(axis=0)/self.probs.shape[0]
        gparams = T.grad(self.cost,self.params)
        pdb.set_trace()
        self.test_func = theano.function([self.v],probs)

if __name__ == '__main__':
    n_visible = 10
    n_hidden = 20
    n_recurrent = 30
    n_components = 2
    test = RNN_RNADE(n_visible,n_hidden,n_recurrent,n_components)
    test.build_RNN_RNADE()
    input_sequence = []
    for i in xrange(100):
        input_sequence.append(numpy.random.random(10))

    probs= test.test_func(input_sequence)
    pdb.set_trace()
    