import os
import sys
import numpy
import theano
import theano.tensor as T
from SGD import SGD_Optimiser
import mocap_data
import pdb

class SGD_mocap(SGD_Optimiser):
    def train(self,valid_set=False,learning_rate=0.1,num_updates=500,save=False,output_folder=None,lr_update=None,
              mom_rate=0.9,update_type='linear',start=2,batch_size=100):
        self.best_cost = numpy.inf
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
        try:
            cost = []
            for u in xrange(num_updates):
                batch_data = mocap_data.sample_train_seq(self.batch_size) #Ensure this is a list in the desired form. 
                pdb.set_trace()
                inputs = [list(batch_data)] + [self.lr]
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
                self.train_costs.append(mean_costs[0])
                if not valid_set:
                    this_cost = numpy.absolute(numpy.mean(cost, axis=0)[0])
                    if this_cost < self.best_cost:
                        self.best_cost = this_cost
                        print 'Best Params!'
                        if save:
                            self.save_model()
                    sys.stdout.flush()     
                else:
                    self.perform_validation()
                
                if self.stop_train:
                    print 'Stopping training early.'
                    break

                if lr_update:
                    self.update_lr(u+1,update_type=self.update_type,start=self.start,num_iterations=self.num_updates)
            print 'Training completed!'

        except KeyboardInterrupt: 
            print 'Training interrupted.'
