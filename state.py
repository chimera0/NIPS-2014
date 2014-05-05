def get_state():
	state = {}
	#model parameters
	state['n_visible'] = 49
	state['n_hidden'] = 50 #{20,40,50,70,80,100} Just guessing here, 
	state['n_recurrent'] = 100 #{40,50,60,80,100} Again guessing
	state['n_components'] = 2 #{2,5,10,20} From RNADE paper
	state['hidden_act'] = 'ReLU'#'sigmoid'
	state['l2'] = 2. #{2.0;1.0;0.1;0.01;0.001;0} From RNADE paper
	#misc parameters
	state['save'] = True
	state['load'] = False
	state['output_folder'] = '/homes/sss31/PhD/NIPS'
	state['load_dir'] = '/homes/sss31/PhD/NIPS'
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
	state['pre_train'] = True #always
	state['rec_mu'] = True
	state['rec_mix'] = True
	state['rec_sigma'] = True
	return state

	#{1;0:05;0:025;0:0125} Try these for pre-training