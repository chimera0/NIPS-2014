"""
Trainer for the mocap dataset for RNN-RNADE
"""
import os
import sys
import numpy
import theano
import theano.tensor as T 
import state
from RNN_RNADE import RNN_RNADE
from SGD_mocap import SGD_mocap
from state import *
import pdb

class trainer:
    def __init__(self,state):
        self.state = state
        for key,val in self.state.iteritems():
            setattr(self,key,val)
        print '*******PARAMS*******'
        for param,value in self.state.iteritems():
            print '%s: %s'%(param,value)
        print '********************'
        self.model = RNN_RNADE(self.n_visible,self.n_hidden,self.n_recurrent,self.n_components,hidden_act=self.hidden_act,l2=self.l2)
        self.model.build_RNN_RNADE()
        self.optmiser = SGD_mocap(self.model.params,[self.model.v],[self.model.cost],momentum=self.momentum,patience=self.patience)

    def train(self,):
        pdb.set_trace()
        self.optmiser.train(learning_rate=self.learning_rate,num_updates=self.num_updates,save=self.save,output_folder=self.output_folder,
                            lr_update=self.lr_update,update_type=self.update_type,mom_rate=self.mom_rate,start=self.start)

if __name__ == '__main__':
    state = get_state()
    trainer_exp = trainer(state)
    trainer_exp.train()