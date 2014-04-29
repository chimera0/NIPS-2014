def get_state():
	state = {}
	#model parameters
	state['n_visible'] = 49
	state['n_hidden'] = 50
	state['n_recurrent'] = 100
	state['n_components'] = 2
	state['hidden_act'] = 'ReLU'#'sigmoid'
	#misc parameters
	state['save'] = True
	state['load'] = False
	state['output_folder'] = '/homes/sss31/datasets/gtzan'
	state['load_dir'] = '/homes/sss31/datasets/gtzan'
	#optimisation parameters
	state['lr_update'] = True
	state['update_type'] = 'linear'
	state['batch_size'] = 100
	state['momentum'] = True
	state['mom_rate'] = 0.9
	state['patience'] = 20
	state['learning_rate'] = 0.1
	state['num_updates'] = 100000
	return state