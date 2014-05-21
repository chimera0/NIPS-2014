import os
import sys
import numpy
import theano
from RNADE import RNADE
from datasets import Dataset
from SGD import SGD_Optimiser
import mocap_data
import matplotlib
matplotlib.use('Agg')
import pylab
import pdb


def plot_costs(optimiser,output_folder,fig_title='Default cost',filename='cost.png',):
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
    filename = os.path.join(output_folder,filename)
    pylab.legend()
    pylab.title(fig_title)
    pylab.savefig(filename)
    if optimiser.valid_costs:
        valid_costs = numpy.array(optimiser.valid_costs)
        if valid_costs.ndim == 1:
            valid_costs = numpy.array(optimiser.valid_costs).reshape(-1)
        else:
            train_costs = numpy.array(optimiser.valid_costs)[:,0]  #0'th cost must be the objective 
        epochs = [i for i in xrange(len(optimiser.valid_costs))]
        pylab.figure()
        pylab.plot(epochs,valid_costs,'r',label='valid loglik')
        pylab.xlabel('epoch')
        pylab.ylabel('negative log-likehihood error.')
        pylab.legend()
        filename = os.path.join(output_folder,'valid_costs.png')
        pylab.title(fig_title)
        pylab.savefig(filename)        


#RNADE params
n_visible = 49
n_hidden = 50
n_recurrent = 100
n_components = 5
print 'Training the RNADE'
l2 = 0.1
hidden_act = 'sigmoid'
rnade = RNADE(n_visible,n_hidden,n_components,hidden_act=hidden_act,l2=l2)
#dataset params
batch_size = 100
num_examples = 100
filename = 'pre_train_params.pickle'
learning_rate = 0.001
train_data = mocap_data.sample_train_seq(batch_size)
for i in xrange(1,num_examples):
    train_data = numpy.vstack((train_data,mocap_data.sample_train_seq(batch_size)))
numpy.random.shuffle(train_data)
#optimiser params
output_folder = '/scratch/Sid/RNADE/50-100-5-0.1-0.001/3'
total_num = train_data.shape[0]
train_frac = 0.8
train_dataset = Dataset([train_data[0:int(train_frac*total_num)]],100)
valid_dataset = Dataset([train_data[int(train_frac*total_num):]],100)
optimiser = SGD_Optimiser(rnade.params,[rnade.v],[rnade.cost,rnade.ll_cost,rnade.l2_cost],momentum=True,patience=20)
optimiser.train(train_dataset,valid_set=valid_dataset,learning_rate=learning_rate,num_epochs=1000,save=True,
            lr_update=True,update_type='linear',start=2,output_folder=output_folder,filename=filename)
plot_costs(optimiser,output_folder,fig_title='Pretraining cost',filename='pretraining.png')
print 'Done pre-training.'
