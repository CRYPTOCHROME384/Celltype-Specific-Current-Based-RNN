import torch as tch 
#from tqdm import trange
import numpy as np
import torch.nn.functional as F





class SingleTrialLicks(tch.nn.Module):
    def __init__(self, params_dict, indexes_neurons_st):
        super(SingleTrialLicks, self).__init__()
        self.device = tch.device(params_dict['device'])
        self.dt = params_dict['dt']
        self.input_noise = params_dict['input_noise']
        self.t_min_input = params_dict['t_min_input']
        self.t_max_input = params_dict['t_max_input']
        self.t_min_test = params_dict['t_max_input']
        self.t_max_test = params_dict['t_max_input'] + 0.15
        self.t_start = params_dict['t_start']
        self.t_go = params_dict['t_go']

        self.is_trial_average = True

        #indexes for neurons that are recorded at single trials
        self.indexes_neurons = indexes_neurons_st

        # left and right
        self.bs = 2 #left and right trials
        self.n_trials = params_dict['n_trials']

        # number of populations
        self.n_pop_alm = params_dict['n_latents'] # number of latents

        # number of neurons
        self.n_neurons_alm = params_dict['n_neurons'] #number of neurons

        # transfer function:
        self.phi = tch.tanh
        #self.phi = tch.sigmoid
    
        # This implementation allows for non-unifrom neuron properties
        self.tau_alm = params_dict['tau'] 

        #other cortex
        self.input_stim = tch.nn.Parameter(tch.rand(self.bs, self.n_pop_alm), requires_grad=True)
        self.input_go = tch.nn.Parameter(tch.rand(self.n_pop_alm), requires_grad=True)
        self.input_licks = tch.nn.Parameter(tch.rand(self.bs, self.n_neurons_alm), requires_grad=True)

         #building network parameters
        self.build_coef_phi()
        self.build_biases()
        self.build_input_trial()
        self.build_decoder()
        self.build_communication_recurrent()

        #perturbing the model
        self.input_test = tch.zeros(self.n_neurons_alm).to(self.device)     
        self.to(self.device)

    def build_coef_phi(self):
        # transfer function:
        self.coef_phi_alm = tch.nn.Parameter(20 * tch.ones(self.n_neurons_alm), requires_grad=True)
        
    def build_biases(self):
        # biases
        amp_biases_alm = .7
        self.biases_alm = tch.nn.Parameter(amp_biases_alm * tch.randn(self.n_neurons_alm), requires_grad=True)
        
    def build_decoder(self, amp_init=1.):
        #decoder ALM left
        amp_decoder_alm = amp_init/np.sqrt(self.n_neurons_alm)
        decoder_init_alm = amp_decoder_alm * tch.randn(self.n_pop_alm, self.n_neurons_alm)
        self.decoder_alm = tch.nn.Parameter(decoder_init_alm, requires_grad=True)  
    
    def build_input_trial(self, amp_init=.1):
        #input trial
        amp_trial = amp_init/np.sqrt(self.n_pop_alm)
        input_trial = amp_trial * tch.randn(self.n_pop_alm, self.n_trials)
        self.input_trial = tch.nn.Parameter(input_trial, requires_grad=True) 

    def build_communication_recurrent(self,  amp_init=.2):
        # Cortico-cortical matrices
        J_rec_init = amp_init * tch.randn(self.n_pop_alm , self.n_pop_alm)
        self.J_rec = tch.nn.Parameter(J_rec_init, requires_grad = True)

    def _input_test(self, t):
        """Input current for the sample period"""
        input_test = self.input_test
        if t > self.t_min_test and t < self.t_max_test:
            input_current = input_test
        else:
            input_current = tch.zeros_like(input_test).to(self.device)
        return input_current

    def _input_sample(self, t):
        """Input current for the sample period
         see Chen et al, Cell, 2024
        """
        input_stim = self.input_stim
        if t > self.t_min_input and t < self.t_min_input + 0.15:
            input_current = input_stim
        elif t > self.t_min_input + 0.25 and t < self.t_min_input + 0.4:
            input_current  = input_stim
        elif t > self.t_min_input + 0.5 and t < self.t_min_input + 0.65:
            input_current =  input_stim
        else:
            input_current = tch.zeros_like(input_stim).to(self.device)
        return input_current
     
    def _input_go(self, t):
        """Input current for the go cue
         see Chen et al, Cell, 2024
        """
        input_go = self.input_go
        if t>self.t_go and t < self.t_go + 0.1:
            input_current  = input_go 
        else:
            input_current = tch.zeros_like(input_go).to(self.device)
        return input_current
    
    def _input_mean_licks(self, t, mean_licks):
        """Input current corresponding
        to the left and right licks,
        motor propioceptive inputs
        """
        input_licks = self.input_licks
        i=int(t/self.dt)
        if  self.t_go + 0.1<t:
            input_current_mean  = tch.einsum('bj,bj->bj', input_licks, mean_licks[:,i,:])
        else:
            input_current_mean = tch.zeros((self.bs, self.n_neurons_alm)).to(self.device)
        return input_current_mean

    def _input_single_trial_licks(self, t, single_trial_licks):
        """Input current corresponding
        to the left and right licks,
        motor propioceptive inputs
        """
        input_licks = self.input_licks
        single_trial_mask = tch.zeros((self.bs, single_trial_licks.shape[1], input_licks.shape[1]), device=self.device)
        for indexes in self.indexes_neurons:
            single_trial_mask[:,:, indexes] = 1
        i=int(t/self.dt)
        if  self.t_go + 0.1<t:
            input_current_single_trial  = tch.einsum('bj,bs->bsj', input_licks, single_trial_licks[:,:, i])
            


        else:
            input_current_single_trial = tch.zeros((self.bs,  single_trial_licks.shape[1], self.n_neurons_alm)).to(input_licks)
        return input_current_single_trial 
    
    def initial_conditions(self):
        #initial condition
        self.u_init_alm = 0.01 * tch.randn(self.n_pop_alm).to(self.device)
        m_alm = tch.stack([self.u_init_alm, self.u_init_alm]).to(self.device)
        m_trials = 0.01 * tch.randn((self.n_trials, self.n_pop_alm)).to(self.device)
        #init rates alm left
        current_inp = self.biases_alm 
        current_rec = m_alm @ (self.J_rec.T @ self.decoder_alm)  
        current_rec_trials = m_trials @ (self.J_rec.T @ self.decoder_alm)
        current = current_rec + current_inp
        current_trials = current_rec_trials + current_inp
        fr_phi_alm =  self.coef_phi_alm * (self.phi(current) + tch.ones_like(current))/2.
        fr_phi_trials = self.coef_phi_alm * (self.phi(current_trials) + tch.ones_like(current_trials))/2.
        init_cond = dict(
            m_alm = m_alm, #mean over trials
            fr_alm = fr_phi_alm,
            h_alm = current,
            m_trials = m_trials,
            fr_trials = fr_phi_trials,
            h_trials = current_trials
        )
        return init_cond
    

    def update_alm_mean(self, r_rec, mean_licks, t):
        """"Update alm mean"""
        #current input
        current_noise = self.input_noise * tch.randn_like(self.biases_alm)
        current_inp = self.biases_alm  + current_noise
        #input test
        current_test = self._input_test(t)
        #current task
        current_task = (self._input_sample(t)+ self._input_go(t)) @ self.decoder_alm
        current_licks = self._input_mean_licks(t, mean_licks) 
        #current recurrent
        overlap = r_rec @ self.decoder_alm.T
        current_rec = overlap @ (self.J_rec.T @ self.decoder_alm) 
        #total current
        current = current_rec + current_inp + current_test + current_task + current_licks
        fr_phi =  self.coef_phi_alm * (self.phi(current) + tch.ones_like(current))/2.
        #euler update
        r_rec = r_rec + self.dt * (-r_rec + fr_phi) / self.tau_alm
        return overlap, current, r_rec

    def update_alm_trials(self, r_rec, t, trial_type, single_trial_licks):
        """"Update alm trial by trial"""
        #current input
        current_noise = self.input_noise * tch.randn_like(self.biases_alm)
        current_inp = self.biases_alm + current_noise
        # current trials
        current_trials =  self.input_trial.T @ self.decoder_alm 
        #input test
        current_test = self._input_test(t)
        #current task
        current_licks = self._input_single_trial_licks(t, single_trial_licks) 
        #left trial recieve left current and right trial right current
        left_trials = tch.zeros(self.n_trials).to(self.device)
        right_trials = tch.zeros(self.n_trials).to(self.device)
        left_trials[trial_type==0] = 1 #left
        right_trials[trial_type==1] = 1 #right
        current_task_l_r = self._input_sample(t) @ self.decoder_alm + self._input_go(t) @ self.decoder_alm
        current_task = tch.einsum('i,l->il', left_trials, current_task_l_r[0])
        current_task += tch.einsum('i,l->il', right_trials, current_task_l_r[1])
        #current recurrent
        overlap = r_rec @ self.decoder_alm.T
        current_rec = overlap @ (self.J_rec.T @ self.decoder_alm) 
        #total current
        current = current_rec + current_inp + current_test + current_task + current_trials + current_licks
        fr_phi =  self.coef_phi_alm * (self.phi(current) + tch.ones_like(current))/2.
        #euler update
        r_rec = r_rec + self.dt * (-r_rec + fr_phi) / self.tau_alm
        return overlap, current, r_rec

    
    def covarainces_decoder(self):
        cov_alm = self.decoder_alm @ self.decoder_alm.T
        cov_decoders = dict(
            cov_alm = cov_alm
        )
        return cov_decoders
        
    def forward(self, mean_licks, single_trial_licks, trial_types):
        bs = mean_licks.shape[0]
        n_times = mean_licks.shape[1]
        #varibales alm left 
        latents_alm = tch.zeros(bs, n_times, self.n_pop_alm).to(self.device)
        rates_alm = tch.zeros(bs, n_times, self.n_neurons_alm).to(self.device)
        currents_alm = tch.zeros(bs, n_times, self.n_neurons_alm).to(self.device)
        #variable trials
        latents_trials = tch.zeros(self.n_trials, n_times, self.n_pop_alm).to(self.device)
        rates_trials = tch.zeros(self.n_trials, n_times, self.n_neurons_alm).to(self.device)
        currents_trials = tch.zeros(self.n_trials, n_times, self.n_neurons_alm).to(self.device)
    
        t = self.t_start
        init_cond = self.initial_conditions()
        m_alm = init_cond['m_alm']
        fr_alm = init_cond['fr_alm']
        h_alm = init_cond['h_alm']
        #trials
        m_trials = init_cond['m_trials']
        fr_trials = init_cond['fr_trials']
        h_trials = init_cond['h_trials']
        for i in range(n_times):
            #store variables alm 
            latents_alm[:,i,:] = m_alm
            rates_alm[:,i,:] = fr_alm
            currents_alm[:,i,:] = h_alm
            latents_trials[:,i,:] = m_trials
            rates_trials[:,i,:] = fr_trials
            currents_trials[:,i,:] = h_trials
            #update variables alm left
            m_alm, h_alm, fr_alm = self.update_alm_mean(fr_alm, mean_licks, t)
            if self.is_trial_average==False:
                #update acording single trials
                m_trials, h_trials, fr_trials = self.update_alm_trials(fr_trials, t, trial_types, single_trial_licks)
            t += self.dt 
        cov_decoder = self.covarainces_decoder()
        results = {'latents_alm': latents_alm,
                    'rates_alm': rates_alm,
                    'currents_alm':currents_alm,
                    'latents_trials': latents_trials,
                    'rates_trials': rates_trials,
                    'currents_trials':currents_trials
                    }
        return results, cov_decoder 

