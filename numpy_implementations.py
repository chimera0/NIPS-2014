#    @profile
import theano
import numpy as np
from utils import *
import pdb
floatX = theano.config.floatX
def logdensity_np(X,model,b_alpha,b_mu,b_sigma):
    """
    X is VxB
    """
    W = model.W.get_value()
    V_alpha = model.V_alpha.get_value()
    V_mu = model.V_mu.get_value()
    V_sigma = model.V_sigma.get_value()
    activation_rescaling = model.activation_rescaling.get_value()

    V,H,C = V_alpha.shape
    V,B = X.shape

    p = np.zeros((B,), dtype=floatX)
    h = np.ndarray((B,H), dtype=floatX)

    alpha = np.ndarray((B,C), dtype=floatX)
    mu    = np.ndarray((B,C), dtype=floatX)
    sigma = np.ndarray((B,C), dtype=floatX)
    oact = np.ndarray((B,C), dtype=floatX)
    temp = np.ndarray((B,), dtype=floatX)
    tempB = np.ndarray((B,), dtype=floatX)
    tempB1 = np.ndarray((B,1), dtype=floatX)
    tempBC = np.ndarray((B,C), dtype=floatX)
    a_tmp = np.ndarray((B,H), dtype=floatX)
    k = 0.5*np.log(2*np.pi)
    for i in xrange(model.n_visible):
       # Compute hidden unit activations
        if i == 0:
            a = np.tile(W[i,:],(B, 1))
        else:
#                a = a + X[[i-1],:].T * W[[i],:]
            np.multiply(X[[i-1],:].T, W[[i],:], out=a_tmp)
            np.add(a, a_tmp, out=a)
       # Compute hidden unit outputs
        np.multiply(a, activation_rescaling[i], out=a_tmp)
        h = sigmoid(a_tmp)
        #np.maximum(a_tmp,0,out=h) # h is BxH
        #Compute mixture components
        #alpha
        alpha = softmax(h.dot(V_alpha[i]) + b_alpha[i]) #BxC
        #np.dot(h, V_alpha[i], out=oact)
        #np.add(oact, b_alpha[i], out=oact)
        #            np.tanh(oact,out=oact)
        #            np.multiply(oact, 10, out=oact)
        #pdb.set_trace()
        #np.max(oact, 1, out=tempB1)
        # tempB1 = oact.max(axis=1)
        # np.subtract(oact, tempB1, out=oact)
        # np.exp(oact, out=oact)
        # np.sum(oact, 1, out=tempB1)
        # np.divide(oact, tempB1, out=alpha)

        #mu
        #            mu = np.dot(h, V_mu[i]) + b_mu[i] #BxC
        np.dot(h, V_mu[i], out=oact)
        np.add(oact, b_mu[i], out=mu)

        #sigma
        sigma = np.exp(h.dot(V_sigma[i]) + b_sigma[i])
        #sigma = np.log(1.0+np.exp((h.dot(V_sigma[i]) + b_sigma[i])*10))/10 #BxC
        # np.dot(h, V_sigma[i], out=oact)
        # np.add(oact, b_sigma[i], out=oact)
        # np.multiply(oact,10, out=oact)
        # np.exp(oact, out=oact)
        # np.add(oact, 1.0, out=oact)
        # np.log(oact, out=oact)
        # np.divide(oact, 10, out=sigma)

        p = p + logsumexp(-0.5 * (((mu - X[[i-1],:].T) / sigma)**2) + np.log(alpha) + (-np.log(sigma)-k))
        # np.subtract(mu, X[[i-1],:].T, out=oact)
        # np.divide(oact, sigma, out=oact)
        # np.power(oact, 2, out=oact)
        # np.multiply(oact, -0.5, out=oact)
        # np.log(alpha, out=alpha)
        # np.add(oact,alpha,out=oact)

        # np.log(sigma, out=tempBC)
        # np.add(tempBC, k, out=tempBC)
        # np.subtract(oact, tempBC, out=oact)

        # tempB1 = np.amax(oact,axis=1)
        # np.subtract(oact, tempB1, out=oact)
        # np.exp(oact, out=oact)
        # np.sum(oact, 1, out=tempB)
        # np.log(tempB, out=tempB)
        # np.add(tempB, tempB1.flatten(), out=tempB)
        # np.add(p, tempB, out=p)
    return p