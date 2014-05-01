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
        self.model = RNN_RNADE(self.n_visible,self.n_hidden,self.n_recurrent,self.n_components,hidden_act=self.hidden_act,l2=self.l2,rec_mu=self.rec_mu,
                               rec_mix=self.rec_mix,rec_sigma=self.rec_sigma,load=self.load,load_dir=self.load_dir)
        self.model.build_RNN_RNADE()
        self.optmiser = SGD_mocap(self.model.params,[self.model.v],[self.model.cost,self.model.log_cost,self.model.l2_cost],momentum=self.momentum,patience=self.patience)

    def train(self,):
        if self.pre_train:
            self.model.init_RNADE()
        self.optmiser.train(learning_rate=self.learning_rate,num_updates=self.num_updates,save=self.save,output_folder=self.output_folder,
                            lr_update=self.lr_update,update_type=self.update_type,mom_rate=self.mom_rate,start=self.start,batch_size=self.batch_size)

    def test(self,):
        self.model = RNN_RNADE(self.n_visible,self.n_hidden,self.n_recurrent,self.n_components,hidden_act=self.hidden_act,l2=self.l2,rec_mu=self.rec_mu,
                               rec_mix=self.rec_mix,rec_sigma=self.rec_sigma,load=self.load,load_dir=self.load_dir)
        self.test_func = theano.function([self.model.v],self.model.log_probs)

if __name__ == '__main__':
    state = get_state()
    trainer_exp = trainer(state)
    trainer_exp.train()