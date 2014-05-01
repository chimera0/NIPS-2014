from RNADE import RNADE
from datasets import Dataset
from SGD import SGD_Optimiser
import mocap_data
import numpy

n_visible = 49
n_hidden = 50
n_components = 2

rnade = RNADE(n_visible,n_hidden,n_components,hidden_act='ReLU',l2=2.)

batch_size = 100
num_examples = 100
train_data = mocap_data.sample_train_seq(batch_size)
for i in xrange(1,num_examples):
    train_data = numpy.vstack((train_data,mocap_data.sample_train_seq(batch_size)))
numpy.random.shuffle(train_data)
train_dataset = Dataset([train_data],100)
optimiser = SGD_Optimiser(rnade.params,[rnade.v],[rnade.cost],momentum=True,patience=500)
optimiser.train(train_dataset,valid_set=None,learning_rate=0.01,num_epochs=1000,save=False,
            lr_update=True,update_type='linear',start=2)
