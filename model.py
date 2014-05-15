import numpy
import cPickle
import os
import sys
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import pdb

class Model:
    def __init__(self,):
        pass

    def load_model(self,load_dir,filename=None):
        """
        This method loads saved parameters to start training at previous checkpoint
        TODO: Currently saving is done by the optimizer, might have to implement 
        that in model class. 
        """
        if not load_dir:
            if not filename:
              load_file = 'best_params.pickle'
            else:
              load_file = filename
            try:
              with open(load_file,'r'):
                print 'Loading saved parameters.'
                init_params = cPickle.load(file('best_params.pickle'))
                for i,j in zip(self.params,init_params):
                  i.set_value(j)
            except IOError:
              print 'There was an error loading parameters from ./best_params.pickle'
        else:
            if not filename:
              path = os.path.join(load_dir,'best_params.pickle')
            else:
              path = os.path.join(load_dir,filename)
            try:
              with open(path,'r'):
                print 'Loading saved parameters from %s.'%(path)
                init_params = cPickle.load(file(path))
                for i,j in zip(self.params,init_params):
                  i.set_value(j)
            except IOError:
              print 'There was an error loading parameters from %s'%(path)
