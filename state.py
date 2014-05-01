def get_state():
	state = {}
	#model parameters
	state['n_visible'] = 49
	state['n_hidden'] = 50
	state['n_recurrent'] = 100
	state['n_components'] = 2
	state['hidden_act'] = 'ReLU'#'sigmoid'
	state['l2'] = 2.
	#misc parameters
	state['save'] = True
	state['load'] = False
	state['output_folder'] = '/homes/sss31/PhD/NIPS'
	state['load_dir'] = '/homes/sss31/PhD/NIPS'
	#optimisation parameters
	state['lr_update'] = True
	state['update_type'] = 'linear'
	state['batch_size'] = 100
	state['momentum'] = True
	state['mom_rate'] = 0.9
	state['patience'] = 20
	state['learning_rate'] = 0.001
	state['num_updates'] = 100000
	state['start'] = 2 #Iteration at which the lr update should start
	state['pre_train'] = False
	state['rec_mu'] = True
	state['rec_mix'] = True
	state['rec_sigma'] = True
	return state