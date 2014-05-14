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
from SGD import SGD_Optimiser
from RNADE import RNADE
from state import *
import pdb
import mocap_data
import matplotlib
matplotlib.use('Agg')
import pylab
from datasets import Dataset

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

    def train(self,):
        if self.pre_train:
            self.pretrain_RNADE()
        print 'Training RNN-RNADE'
        self.optmiser = SGD_mocap(self.model.params,[self.model.v],[self.model.cost,self.model.neg_ll_cost,self.model.l2_cost],momentum=self.momentum,patience=self.patience)
        self.optmiser.train(learning_rate=self.learning_rate,num_updates=self.num_updates,save=self.save,output_folder=self.output_folder,
                            lr_update=self.lr_update,update_type=self.update_type,mom_rate=self.mom_rate,start=self.start,batch_size=self.batch_size)

    def test(self,):
        self.model = RNN_RNADE(self.n_visible,self.n_hidden,self.n_recurrent,self.n_components,hidden_act=self.hidden_act,l2=self.l2,rec_mu=self.rec_mu,
                               rec_mix=self.rec_mix,rec_sigma=self.rec_sigma,load=self.load,load_dir=self.load_dir)
        self.test_func = theano.function([self.model.v],self.model.log_probs)

    def pretrain_RNADE(self,):
        print 'Pre-training the RNADE'
        rnade = RNADE(self.n_visible,self.n_hidden,self.n_components,hidden_act=self.hidden_act,l2=self.l2)
        batch_size = 100
        num_examples = 100
        train_data = mocap_data.sample_train_seq(batch_size)
        for i in xrange(1,num_examples):
            train_data = numpy.vstack((train_data,mocap_data.sample_train_seq(batch_size)))
        numpy.random.shuffle(train_data)
        total_num = train_data.shape[0]
        train_frac = 0.8
        train_dataset = Dataset([train_data[0:int(train_frac*total_num)]],100)
        valid_dataset = Dataset([train_data[int(train_frac*total_num):]],100)
        optimiser = SGD_Optimiser(rnade.params,[rnade.v],[rnade.cost],momentum=True,patience=20)
        optimiser.train(train_dataset,valid_set=valid_dataset,learning_rate=0.001,num_epochs=500,save=False,
                    lr_update=True,update_type='linear',start=2)
        self.plot_costs(optimiser,fig_title='Pretraining cost',filename='pretraining.png')
        for param in rnade.params:
            value = param.get_value()
            self.model.params_dict[param.name].set_value(value)
        print 'Done pre-training.'

    def test(self,):
        self.model = RNN_RNADE(self.n_visible,self.n_hidden,self.n_recurrent,self.n_components,hidden_act=self.hidden_act,l2=self.l2,rec_mu=self.rec_mu,
                               rec_mix=self.rec_mix,rec_sigma=self.rec_sigma,load=True,load_dir=self.load_dir)
        num_test_sequences = 100
        batch_size = 100
        num_samples = 10
        error = []
        for i in xrange(num_test_sequences):
            seq = mocap_data.sample_train_seq(batch_size) 
            samples = self.model.sample_given_sequence(seq,num_samples)
            error.append(numpy.mean((samples - seq)**2))    
        total_error = numpy.mean(error)
        print 'The squared prediction error per frame per sequence is: %f'%(total_error)

    def plot_costs(self,optimiser,fig_title='Default cost',filename='cost.png'):
        epochs = [i for i in xrange(len(optimiser.train_costs))]
        train_costs = numpy.array(optimiser.train_costs).reshape(-1)
        pylab.plot(epochs,train_costs,'b',label='training ll')
        pylab.xlabel('epoch')
        pylab.ylabel('negative log-likelihood')
        filename = os.path.join(self.output_folder,filename)
        if optimiser.valid_costs is not None:
            valid_costs = numpy.array(optimiser.valid_costs).reshape(-1)
            pylab.plot(epochs,valid_costs,'r',label='valid ll')
            pylab.savefig(filename)
        else:
            pylab.savefig(filename)
        
        
        pylab.title(fig_title)
        pylab.savefig(filename)


if __name__ == '__main__':
    state = get_state()
    trainer_exp = trainer(state)
    trainer_exp.train()
    trainer_exp.test()