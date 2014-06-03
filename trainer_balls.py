"""
Trainer for the mocap dataset for RNN-RNADE
"""
import os
import sys
import numpy
import cPickle
import theano
import theano.tensor as T 
import state
from RNN_RNADE import RNN_RNADE
from SGD_balls import SGD_balls
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
        if self.load_pretrain:
            self.load_pretrain_RNADE()
        self.results = {}
        self.model.build_RNN_RNADE()

    def load_pretrain_RNADE(self,):
        rnade = RNADE(self.n_visible,self.n_hidden,self.n_components,hidden_act=self.hidden_act,l2=self.l2)
        rnade.load_model(self.pretrain_folder,'pre_train_params.pickle')
        for param in rnade.params:
            value = param.get_value()
            self.model.params_dict[param.name].set_value(value)

    def train(self,):
        if self.pre_train:
            self.pretrain_RNADE()
        print 'Training RNN-RNADE'
        self.optimiser = SGD_balls(self.model.params,[self.model.v],[self.model.cost,self.model.neg_ll_cost,self.model.l2_cost],
                                   momentum=self.momentum,patience=self.patience,state=self.state,clip_gradients=self.clip_gradients,
                                   grad_threshold=self.grad_threshold)
        self.optimiser.train(learning_rate=self.learning_rate,num_updates=self.num_updates,save=self.save,output_folder=self.output_folder,
                            lr_update=self.lr_update,update_type=self.update_type,mom_rate=self.mom_rate,start=self.start,batch_size=self.batch_size)
        optimiser = self.optimiser
        self.plot_costs(optimiser,fig_title='RNN-RNADE Training Cost',filename='training_cost.png')
        self.results['train_costs'] = self.optimiser.train_costs
        cPickle.dump(self.results,open(os.path.join(self.output_folder,'results.pickle'),'w'))
        #self.results['valid_costs'] = self.optimiser.valid_costs

    # def test(self,):
    #     self.model = RNN_RNADE(self.n_visible,self.n_hidden,self.n_recurrent,self.n_components,hidden_act=self.hidden_act,l2=self.l2,rec_mu=self.rec_mu,
    #                            rec_mix=self.rec_mix,rec_sigma=self.rec_sigma,load=self.load,load_dir=self.load_dir)
    #     self.test_func = theano.function([self.model.v],self.model.log_probs)

    def pretrain_RNADE(self,):
        print 'Pre-training the RNADE'
        l2 = 2.
        rnade = RNADE(self.n_visible,self.n_hidden,self.n_components,hidden_act=self.hidden_act,l2=l2)
        batch_size = 100
        num_examples = 100
        filename = 'pre_train_params.pickle'
        learning_rate = self.learning_rate_pretrain
        train_data = mocap_data.sample_train_seq(batch_size)
        for i in xrange(1,num_examples):
            train_data = numpy.vstack((train_data,mocap_data.sample_train_seq(batch_size)))
        numpy.random.shuffle(train_data)
        total_num = train_data.shape[0]
        train_frac = 0.8
        train_dataset = Dataset([train_data[0:int(train_frac*total_num)]],100)
        valid_dataset = Dataset([train_data[int(train_frac*total_num):]],100)
        optimiser = SGD_Optimiser(rnade.params,[rnade.v],[rnade.cost,rnade.ll_cost,rnade.l2_cost],momentum=True,patience=20,clip_gradients=self.clip_gradients)
        optimiser.train(train_dataset,valid_set=valid_dataset,learning_rate=learning_rate,num_epochs=5,save=True,
                    lr_update=True,update_type='linear',start=2,output_folder=self.output_folder,filename=filename)
        self.plot_costs(optimiser,fig_title='Pretraining cost',filename='pretraining.png')
        print 'Done pre-training.'
        ####load best params from pre-training###
        print 'Loading best RNADE parameters'
        rnade = RNADE(self.n_visible,self.n_hidden,self.n_components,hidden_act=self.hidden_act,l2=l2)
        rnade.load_model(self.output_folder,filename=filename)
        ###########
        for param in rnade.params:
            value = param.get_value()
            self.model.params_dict[param.name].set_value(value)
        print 'Done pre-training.'
        #Saving results to dict
        self.results['pretraining_train_costs'] = optimiser.train_costs
        self.results['pretraining_valid_costs'] = optimiser.valid_costs

    def test(self,):
        self.model = RNN_RNADE(self.n_visible,self.n_hidden,self.n_recurrent,self.n_components,hidden_act=self.hidden_act,l2=self.l2,rec_mu=self.rec_mu,
                               rec_mix=self.rec_mix,rec_sigma=self.rec_sigma,load=True,load_dir=self.load_dir)
        num_test_sequences = 100
        batch_size = 100
        num_samples = 1
        error = []
        for i in xrange(num_test_sequences):
            seq = mocap_data.sample_test_seq(batch_size) 
            samples = self.model.sample_given_sequence(seq,num_samples)
            sq_diff = (samples - seq)**2
            sq_diff = sq_diff.mean(axis=0)
            sq_diff = sq_diff.sum(axis=1)
            seq_error = sq_diff.mean(axis=0)   
            error.append(seq_error)
        total_error = numpy.mean(error)
        self.results['error_list'] = error
        self.results['mean_error'] = total_error
        cPickle.dump(self.results,open(os.path.join(self.output_folder,'results.pickle'),'w'))
        print 'The squared prediction error per frame per sequence is: %f'%(total_error)

    def plot_costs(self,optimiser,fig_title='Default cost',filename='cost.png'):
        epochs = [i for i in xrange(len(optimiser.train_costs))]
        train_costs = numpy.array(optimiser.train_costs)
        if train_costs.ndim == 1:
            train_costs = numpy.array(optimiser.train_costs).reshape(-1)
        else:
            train_costs = numpy.array(optimiser.train_costs)[:,0]  #0'th cost must be the objective 

        pylab.figure()
        pylab.plot(epochs,train_costs,'b',label='training loglik')
        pylab.xlabel('epoch')
        pylab.ylabel('negative log-likelihood')
        filename = os.path.join(self.output_folder,filename)
        pylab.legend()
        pylab.title(fig_title)
        pylab.savefig(filename)
        if optimiser.valid_costs:
            valid_costs = numpy.array(optimiser.valid_costs)
            if valid_costs.ndim == 1:
                valid_costs = numpy.array(optimiser.valid_costs).reshape(-1)
            else:
                valid_costs = numpy.array(optimiser.valid_costs)[:,0]  #0'th cost must be the objective 
            epochs = [i for i in xrange(len(optimiser.valid_costs))]
            pylab.figure()
            pylab.plot(epochs,valid_costs,'b',label='sq pred. error')
            pylab.xlabel('epoch')
            pylab.ylabel('squared_prediction error')
            pylab.legend()
            filename = os.path.join(self.output_folder,'valid_costs_full.png')
            pylab.title(fig_title)
            pylab.savefig(filename)        

if __name__ == '__main__':
    state = get_state_balls()
    # args = sys.argv
    # n_hidden=int(args[1])
    # n_recurrent=int(args[2])
    # n_components=int(args[3])
    # l2=float(args[4])
    # learning_rate=float(args[5])
    # learning_rate_pretrain=float(args[6])
    # trial = int(args[7])
    # name = "nh-{0}_nr-{1}_nc-{2}_l2-{3}_lr-{4}_lrpt-{5}_trial-{6}".format(n_hidden,n_recurrent,n_components,l2,learning_rate,learning_rate_pretrain,trial)
    # output_folder = create_output_folder(name)
    # state = get_state_args(n_hidden=n_hidden,
    #                        n_recurrent=n_recurrent,
    #                        n_components=n_components,
    #                        l2=l2,
    #                        learning_rate=learning_rate,
    #                        learning_rate_pretrain=learning_rate_pretrain,
    #                        output_folder = output_folder)
    trainer_exp = trainer(state)
    trainer_exp.train()
    #trainer_exp.test()