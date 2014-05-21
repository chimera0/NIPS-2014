import os
import sys
import numpy
import theano
import theano.tensor as T
from SGD import SGD_Optimiser
from RNN_RNADE import RNN_RNADE
import mocap_data
import pdb

class SGD_mocap(SGD_Optimiser):
    def train(self,valid_set=False,learning_rate=0.1,num_updates=500,save=False,output_folder=None,lr_update=None,
              mom_rate=0.9,update_type='linear',start=2,batch_size=100,filename=None):
        self.best_train_cost = numpy.inf
        self.best_valid_cost = numpy.inf
        self.init_lr = learning_rate
        self.lr = numpy.array(learning_rate)
        self.mom_rate = mom_rate
        self.output_folder = output_folder
        self.valid_set = valid_set
        self.save = save
        self.lr_update = lr_update
        self.stop_train = False
        self.train_costs = []
        self.valid_costs = []
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.update_type = update_type
        self.start = start
        self.filename = filename
        self.valid_frequency = 1000
        self.save_model() #saving model after pre-training
        try:
            cost = []
            for u in xrange(num_updates):
                #pdb.set_trace()
                if u%1000 == 0:
                    self.valid()
                else:
                    batch_data = mocap_data.sample_train_seq(self.batch_size) #Ensure this is a list in the desired form. 
                    fixed_array = numpy.zeros(batch_data.shape)
                    fixed_array[:] = batch_data
                    inputs = [fixed_array] + [self.lr]
                    if self.momentum:
                        inputs = inputs + [self.mom_rate]
                    cost.append(self.f(*inputs))
                    mean_costs = numpy.mean(cost,axis=0)
                    if numpy.isnan(mean_costs[0]):
                        print 'Training cost is NaN.'
                        print 'Breaking from training early, the last saved set of parameters is still usable!'
                        break
                    print '  Update %i   ' %(u+1)
                    print '***Train Results***'
                    for i in xrange(self.num_costs):
                        print "Cost %i: %f"%(i,mean_costs[i])
                    self.train_costs.append(mean_costs)
                    this_cost = numpy.absolute(numpy.mean(cost, axis=0)[0])
                    if this_cost < self.best_train_cost:
                        self.best_train_cost = this_cost
                        print 'Best Params!'
                        if save:
                            self.save_model('best_params_train.pickle')
                    sys.stdout.flush()     
             
                    if self.stop_train:
                        print 'Stopping training early.'
                        break

                    if lr_update:
                        self.update_lr(u+1,update_type=self.update_type,start=self.start,num_iterations=self.num_updates)
            print 'Training completed!'

        except KeyboardInterrupt: 
            print 'Training interrupted.'

    def valid(self,):
        print 'Performing validation.'
        model = RNN_RNADE(self.state['n_visible'],self.state['n_hidden'],self.state['n_recurrent'],self.state['n_components'],hidden_act=self.state['hidden_act'],
                l2=self.state['l2'],rec_mu=self.state['rec_mu'],rec_mix=self.state['rec_mix'],rec_sigma=self.state['rec_sigma'],load=False,load_dir=self.output_folder)
        model.params = self.params
        #model.load_model(self.output_folder,'best_params_train.pickle')
        num_test_sequences = 10
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
        print 'Validation error: ',total_error
        self.valid_costs.append(total_error)
        if total_error < self.best_valid_cost:
            print 'Best validation params!'
            self.best_valid_cost = total_error
            self.save_model('best_params_valid.pickle')