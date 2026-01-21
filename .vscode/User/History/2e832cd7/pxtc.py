import json 
import torch as tch
from losses import *
from model import SingleTrialRateInputLicks
from plotting import *
from utils import *
import numpy as np
import time
import pickle
import random


path_save = '../results/'

#loading hiperparameters data
with open('parameters_list.json') as f:
    default_parameters = json.load(f)


def train(files_list, n_pop=15, net_params=None, lr=5e-4, max_epochs = 500000, total_trials = 300, info = ''):

    #coeficient losses
    c_coding_av = .1 #rec neurons
    c_trials_av = 0.33 #rec av rates left
    c_time_av = 0.25
    c_orth = 500
    c_sel = 0.5
    test_every = 50
    optimize_every = 1
    #loading data    
    mean_licks = mean_licks_inputs(files_list, hemi = hemi, max_time_index = max_time_index) 
    single_trial_licks = licks_inputs(files_list, max_time_index =  max_time_index)
    ALM_hemi= load_me_data(files_list, default_parameters, hemi=hemi, max_time_index=max_time_index)
    trial_average = tch.tensor(ALM_hemi['rates'][0:2,:,:], dtype = tch.float32).to(default_parameters['device']).float()
    print('dimensions', hemi, trial_average.shape, mean_licks.shape)


    #trial average
    LossTrials = LossAverageTrials()
    losses_trial_av = np.zeros(max_epochs)
    #loss on the modes
    LossSelectivity = SelectivityLoss(trial_average, default_parameters)
    losses_sel = np.zeros(max_epochs)
    #loss on the orthogonality of the weights
    LossWeights = LossOrthogonality(default_parameters['device']) 
    losses_weights = np.zeros(max_epochs)
    losses_time_av = np.zeros(max_epochs)
    losses_neurons_av = np.zeros(max_epochs)
    #Defining losses
    #average over time
    ListLossesTime = [LossAverageTime(ALM_hemi['indexes'][s], default_parameters) for s in range(total_sessions)]
    #average over neurons
    ListLossesNeurons = [LossAverageNeurons(ALM_hemi['indexes'][s]) for s in range(total_sessions)]

    #hyperparameters
    test_every = 50
    optimize_every = 1

    #first trial
    default_parameters['n_trials'] = len(trial_types)
    net = SingleTrialRate(default_parameters)
    net.device = default_parameters['device']
    if net_params is not None:
        try:
            net.load_state_dict(tch.load(net_params))
        except:
            print('Different network sizes')
    optimizer = tch.optim.Adam(net.parameters(), lr=lr)
    epoch=0
    min_mean_loss = 10. 
    thres_trial_av = 10.
    mean_loss = 100.
    while epoch<max_epochs:
        #adding initial data
        results, cov = net.forward(trial_average, trial_types)
 
        #Reconsutring neural activity average over trials
        total_trial_average = LossTrials(trial_average, results['rates_alm'])
        losses_trial_av[epoch] = total_trial_average.detach().cpu().numpy()

        #Reconsutring average activity over neurons
        ind_trial = 0
        total_average_neurons = 0
        for s in range(total_sessions):
            total_average_neurons += ListLossesNeurons[s](ALM_hemi['neurons_average'][s][trial_indexes[s]], results['rates_trials'][ind_trial:ind_trial+d_ind])
            ind_trial+=d_ind
        total_average_neurons = total_average_neurons/total_sessions
        losses_neurons_av[epoch] = total_average_neurons.detach().cpu().numpy()

        #Reconsutring average activity over time
        total_time_average = 0
        ind_trial = 0
        total_average_neurons = 0
        for s in range(total_sessions):
            #trial indexes to compute
            total_time_average += ListLossesTime[s](ALM_hemi['time_average'][s][:,trial_indexes[s],:], results['rates_trials'][ind_trial:ind_trial+d_ind])
            ind_trial+=d_ind
        total_time_average = total_time_average/total_sessions
        losses_time_av[epoch] = total_time_average .detach().cpu().numpy()
        
        #Loss orthogonality factors
        total_orth = LossWeights(cov)
        losses_weights[epoch] = total_orth.detach().cpu().numpy()

        #Loss selectivity
        total_selectivity = LossSelectivity(net)
        losses_sel[epoch] = total_selectivity.detach().cpu().numpy()

        #defining total loss
        loss_factors =  c_orth * total_orth + c_sel * total_selectivity
        
        if thres_trial_av<mean_loss:
            loss_reconstruction = total_trial_average
        else:
            loss_reconstruction = c_neurons_av * total_average_neurons + c_trials_av * total_trial_average + c_time_av * total_time_average
        total_loss = 0.7 * loss_reconstruction + 0.3 * loss_factors
        total_loss= total_loss/ optimize_every

        #mean reconstruction loss
        mean_trial = np.mean(losses_trial_av[max(0, epoch-100):epoch])
        mean_loss = mean_trial
        mean_time = np.mean(losses_time_av[max(0, epoch-100):epoch])
        mean_neurons = np.mean(losses_neurons_av[max(0, epoch-100):epoch])
        sel_loss = np.mean(losses_sel[max(0, epoch-100):epoch])
        weights_loss = np.mean(losses_weights[max(0, epoch-100):epoch])
        print(info+f'Epoch {epoch}'+f' | Trial Average = {mean_trial}'+f' | Time Average = {mean_time}'+f' | Neurons Average = {mean_neurons}'+ f'| Selectivity = {sel_loss}'+f' | Orthogonality = {np.mean(weights_loss)}')
        
        #optimizing
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
            
        epoch += 1
        if epoch != 0 and epoch %test_every == 0 and mean_loss < min_mean_loss:
            min_mean_loss = min(min_mean_loss, mean_trial)
            print('saving net')
            name_f = path_save+'net_ALM_'+str(n_pop)+ '.pth'
            tch.save(net.state_dict(), name_f)   
        if epoch != 0 and epoch % test_every == 0:
            #Trial average
            name_file = path_save+ 'trial_average_npop_'+str(n_pop)
            name_loss = 'Loss Reconstructions Neurons'
            pickle.dump(losses_trial_av, open(name_file+'.p', 'wb'))
            loss_plot = losses_trial_av
            plot_loss(epoch, loss_plot, name_loss, tag = name_file)
            #Time average
            name_file = path_save+ 'time_average_npop_'+str(n_pop)
            name_loss = 'Time Loss Reconstructions Neurons'
            pickle.dump(losses_time_av, open(name_file+'.p', 'wb'))
            loss_plot = losses_time_av
            plot_loss(epoch, loss_plot, name_loss, tag = name_file)
            #Neuron average
            name_file = path_save+ 'neuron_average_npop_'+str(n_pop)
            name_loss = 'Neuron Loss Reconstructions Neurons'
            pickle.dump(losses_neurons_av, open(name_file+'.p', 'wb'))
            loss_plot = losses_neurons_av
            plot_loss(epoch, loss_plot, name_loss, tag = name_file)
            # weights and selectivity
            name_file = path_save+'weights_npop_'+str(n_pop)
            pickle.dump(losses_weights, open(name_file+'.p', 'wb'))
            name_file = path_save+'selectivity_npop_'+str(n_pop)
            pickle.dump(losses_sel, open(name_file+'.p', 'wb'))
 
