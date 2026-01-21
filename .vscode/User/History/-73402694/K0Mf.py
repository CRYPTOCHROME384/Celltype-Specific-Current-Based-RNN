import numpy as np
import os
import pickle
import torch as tch 


#path_dandi = '../data/MAPDANDI/'
path_dandi = '../data/MAPDANDI_Functional/'

def hemi_indexes(data, x_hemi = 5600):
    """
    ccf_x   :  int   # (um)  Left-to-Right (ML axis) --> v axis
    threshold ccf_x 5600
    """
    ml = np.array([ccf[0] for ccf in data['ccf']])
    indexes_right = ml>x_hemi
    indexes_left = ml<=x_hemi
    return indexes_left, indexes_right

def mean_licks_inputs(list_sessions, hemi = 'left', max_time_index = 350, ind_start=50):
    trial_average, indexes = trial_average_all_neurons(list_sessions, hemi = hemi, max_time_index= max_time_index)
    trial_average = tch.tensor(trial_average, dtype = tch.float32).to('cuda').float()
    mean_licks =  tch.zeros_like(trial_average)
    ind_ses = 0
    for file in list_sessions:
        data = pickle.load(open(path_dandi + file,'rb'))
        indexes_left, indexes_right = hemi_indexes(data)
        if hemi=='left':
            indexes_neurons = indexes_left
        else:
            indexes_neurons = indexes_right
        data_bh = pickle.load(open(path_dandi+ file.rstrip('.p') + '_behavior.p', 'rb'))
        m_left_licks = np.mean(data_bh['rates_left_licks'][(data_bh['trial_type']==0) * (data_bh['trial_responses']=='hit'),ind_start:max_time_index + ind_start], axis = 0)
        m_right_licks = np.mean(data_bh['rates_right_licks'][(data_bh['trial_type']==1) * (data_bh['trial_responses']=='hit'),ind_start:max_time_index + ind_start], axis = 0)
        for l in range(ind_ses,ind_ses+np.sum(indexes_neurons)):
            mean_licks[0, :, l] = tch.tensor(m_left_licks, dtype = tch.float32, device = 'cuda')
            mean_licks[1, :, l] = tch.tensor(m_right_licks,dtype = tch.float32, device = 'cuda')
        ind_ses+=np.sum(indexes_neurons)
    return mean_licks

def licks_inputs(list_sessions, hemi = 'left', max_time_index = 350, ind_start=50):
    ind_ses = 0
    for file in list_sessions:
        data_bh = pickle.load(open(path_dandi+ file.rstrip('.p') + '_behavior.p', 'rb'))
        left_licks = data_bh['rates_left_licks'][:,ind_start:max_time_index + ind_start]
        right_licks = data_bh['rates_left_licks'][:,ind_start:max_time_index + ind_start]
        sess_licks = np.concatenate((m_left_licks,right_licks)) 
        sess_licks = tch.tensor(sess_licks, dtype = tch.float32, device = 'cuda')
    return mean_licks

def trial_average_all_neurons(list_sessions, hemi = 'left', max_time_index = 350):
    mean_rates = []
    indexes_list = []
    s = 0
    for file in list_sessions:
        data = pickle.load(open(path_dandi + file,'rb'))
        indexes_left, indexes_right = hemi_indexes(data)
        if hemi=='left':
            indexes_neurons = indexes_left
        else:
            indexes_neurons = indexes_right
        rates = np.array([data['average_trials_left'][0:max_time_index, indexes_neurons], data['average_trials_right'][0:max_time_index, indexes_neurons]]) 
        mean_rates.append(rates)
        indexes_list.append(range(s, s + data['average_trials'].shape[1]))
        s+=data['average_trials'][:,indexes_neurons].shape[1]
    if len(list_sessions)>1:
        trial_average = np.concatenate(mean_rates, axis=2)
    else:
        trial_average = rates 
    return trial_average, indexes_list

def time_averages(list_sessions, default_parameters, hemi = 'left'):
    time_averages = []
    for file in list_sessions:
        data = pickle.load(open(path_dandi + file,'rb'))
        indexes_left, indexes_right = hemi_indexes(data)
        if hemi=='left':
            indexes_neurons = indexes_left
        else:
            indexes_neurons = indexes_right
        t_average = np.einsum('ijk->kij', data['average_time'][:,indexes_neurons,:]) #neuron,condition, trial->condition, trial, neuron
        t_average = tch.tensor(t_average, dtype = tch.float32).to(default_parameters['device']).float()
        time_averages.append(t_average)
    return time_averages

def neurons_averages(list_sessions, default_parameters, hemi = 'left'):
    neurons_averages = []
    for file in list_sessions:
        data = pickle.load(open(path_dandi+file,'rb'))
        n_average = data['average_neurons']#np.einsum('ijk->jki', data['neurons_average'])
        n_average = tch.tensor(n_average, dtype = tch.float32).to(default_parameters['device']).float()
        neurons_averages.append(n_average)
    return neurons_averages


def number_of_trials(list_sessions):
    """List of number of left and right trials"""
    num_trials = []
    for file in list_sessions:
        data = pickle.load(open(path_dandi + file,'rb'))
        n_trials = len(data['trial_type'])
        num_trials.append(n_trials)
    num_trials = np.array(num_trials)
    return num_trials

def lists_of_trial_types(list_sessions):
    """List of number of left and right trials"""
    trial_types = []
    for file in list_sessions:
        data = pickle.load(open(path_dandi + file,'rb'))
        trial_types.append(data['trial_type'])
    return trial_types

def lists_of_responses(list_sessions):
    """List of number of left and right trials"""
    trial_responses = []
    for file in list_sessions:
        data = pickle.load(open(path_dandi + file,'rb'))
        trial_responses.append(data['trial_responses'])
    return trial_responses

def load_me_data(list_sessions, default_parameters, hemi = 'left', max_time_index=350):
    """
    rates: trial average firing rates for correct left and right
    indexes: indexes of neurons for each session
    n_trials: number of trials per session
    trial_types: left or right trials
    time_average: the average over some epochs for each trial
    neuron average: the average over the entire recorded neurons at this session (left and right hemi)
    responses: 'hit, miss, ignore'
    """
    trial_average, indexes = trial_average_all_neurons(list_sessions, hemi = hemi, max_time_index= max_time_index)
    #List of number of trials
    n_trials = number_of_trials(list_sessions)
    trial_types = lists_of_trial_types(list_sessions)
    time_av = time_averages(list_sessions, default_parameters, hemi = hemi)
    neurons_av = neurons_averages(list_sessions, default_parameters)
    responses = lists_of_responses(list_sessions)
    ALM_data = {
        'rates':trial_average,
        'indexes':indexes,
        'n_trials':n_trials,
        'trial_types':trial_types,
        'time_average': time_av,
        'neurons_average':neurons_av,
        'responses':responses
    }
    return ALM_data



def toch_version_ratates_trial_types(ALM_hemi, default_parameters, total_trials):
    #only correct trials for training
    rates = tch.tensor(ALM_hemi['rates'][0:2,:,:], dtype = tch.float32).to(default_parameters['device']).float()
    # behavior_lick_times - 0: left lick; 1: right lick
    total_sessions = len(ALM_hemi['trial_types'])
    trial_types = []
    for x in ALM_hemi['trial_types']:
        trial_types+=x[10:int(total_trials/total_sessions)+10]
    #trial_types[ALM_hemi['trial_types'][session_number]=='r']=1
    trial_types = tch.tensor(trial_types, dtype = tch.float32).to(default_parameters['device']).float()
    return rates, trial_types