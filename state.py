

import os
def ensure_dir(f):
    print "ensuring:",f
    # d = os.path.dirname(f)
    d = f
    if not os.path.exists(d):
        print "making:",d
        os.makedirs(d)
    else:
        print d,"exists"

def get_state():
    state = {}
    #model parameters
    state['n_visible'] = 49
    state['n_hidden'] = 100 #{20,40,50,70,80,100} Just guessing here, 
    state['n_recurrent'] = 300 #{40,50,60,80,100} Again guessing
    state['n_components'] = 2 #{2,5,10,20} From RNADE paper1
    state['hidden_act'] = 'sigmoid'#'sigmoid'
    state['l2'] = 0.1 #{2.0;1.0;0.1;0.01;0.001;0} From RNADE paper
    #misc parameters
    state['save'] = True
    state['load'] = False
    state['output_folder'] = '/scratch/Sid/RNN_RNADE_2/test'
    state['load_dir'] = '/scratch/Sid/RNN_RNADE_2/test'
    #optimisation parameters
    state['lr_update'] = True
    state['update_type'] = 'linear' #This is essential for the RNADE. Not sure if we need it for the joint model. We can experiment with 'linear' or False
    state['batch_size'] = 100
    state['momentum'] = True
    state['mom_rate'] = 0.9
    state['patience'] = 20
    state['learning_rate'] = 0.001  #{0.1,.01,.001,0.0001,0.0005} Guessing. Should try only small values for this.
    state['num_updates'] = 100000
    state['start'] = 2 #Iteration at which the lr update should start
    state['pre_train'] = False #always
    state['rec_mu'] = True
    state['rec_mix'] = False
    state['rec_sigma'] = False
    state['load_pretrain'] = True
    state['pretrain_folder'] = '/scratch/Sid/RNADE/100-300-2-0.1-0.001/3'
    state['learning_rate_pretrain'] = 0.001
    return state

#{1;0:05;0:025;0:0125} Try these for pre-training

def create_output_folder(name):
    start = os.getcwd()
    fname = os.path.join(start,"results",name)
    print "creating dir:",fname
    ensure_dir(fname)
    return fname

def get_state_args(n_hidden=50,
               n_recurrent=100,
               n_components=2,
               l2=2.,
               learning_rate=0.001,
               learning_rate_pretrain=0.001,
               output_folder = ""
               ):
    state = {}
    #model parameters
    state['n_visible'] = 49
    state['n_hidden'] = n_hidden #{20,40,50,70,80,100} Just guessing here, 
    state['n_recurrent'] = n_recurrent #{40,50,60,80,100} Again guessing
    state['n_components'] = n_components #{2,5,10,20} From RNADE paper
    state['hidden_act'] = 'ReLU'#'sigmoid'
    state['l2'] = l2 #{2.0;1.0;0.1;0.01;0.001;0} From RNADE paper
    #misc parameters
    state['save'] = True
    state['load'] = False
    state['output_folder'] = output_folder
    state['load_dir'] = output_folder
    #optimisation parameters
    state['lr_update'] = True
    state['update_type'] = 'linear' #This is essential for the RNADE. Not sure if we need it for the joint model. We can experiment with 'linear' or False
    state['batch_size'] = 100
    state['momentum'] = True
    state['mom_rate'] = 0.9
    state['patience'] = 20
    state['learning_rate'] = learning_rate  #{0.1,.01,.001,0.0001,0.0005} Guessing. Should try only small values for this.
    state['num_updates'] = 100000
    state['start'] = 2 #Iteration at which the lr update should start
    state['pre_train'] = True #always
    state['learning_rate_pretrain'] = learning_rate_pretrain
    state['rec_mu'] = True
    state['rec_mix'] = True
    state['rec_sigma'] = True
    return state

#{1;0:05;0:025;0:0125} Try these for pre-training



# def get_state():
#   state = {}
#   #model parameters
#   state['n_visible'] = 49
#   state['n_hidden'] = 50 #{20,40,50,70,80,100} Just guessing here, 
#   state['n_recurrent'] = 100 #{40,50,60,80,100} Again guessing
#   state['n_components'] = 2 #{2,5,10,20} From RNADE papertest_outputs
#   state['hidden_act'] = 'sigmoid'#'sigmoid'
#   state['l2'] = 0.001 #{2.0;1.0;0.1;0.01;0.001;0} From RNADE paper
#   #misc parameters
#   state['save'] = True
#   state['load'] = False
#   state['output_folder'] = '/homes/sss31/PhD/NIPS/test_outputs'
#   state['load_dir'] = '/homes/sss31/PhD/NIPS/test_outputs'
#   #optimisation parameters
#   state['lr_update'] = True
#   state['update_type'] = 'linear' #This is essential for the RNADE. Not sure if we need it for the joint model. We can experiment with 'linear' or False
#   state['batch_size'] = 100
#   state['momentum'] = True
#   state['mom_rate'] = 0.9
#   state['patience'] = 20
#   state['learning_rate'] = 0.001  #{0.1,.01,.001,0.0001,0.0005} Guessing. Should try only small values for this.
#   state['num_updates'] = 5
#   state['start'] = 2 #Iteration at which the lr update should start
#   state['pre_train'] = False #always
#   state['rec_mu'] = True
#   state['rec_mix'] = True
#   state['rec_sigma'] = True
#   return state

#   #{1;0:05;0:025;0:0125} Try these for pre-training
