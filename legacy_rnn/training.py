import json 
import torch as tch
from losses import *
from model import  SingleTrialLicks 
from plotting import *
from utils_alm import *
import numpy as np
import time
import pickle
import random
from sklearn.model_selection import train_test_split

path_save = '../results/'

#loading hiperparameters data
with open('parameters_list.json') as f:
    default_parameters = json.load(f)


def train(files_list, session_names = [], net_params=None, n_pop=15, hemi='right', max_time_index =275, lr=5e-4, max_epochs = 300000, test_every = 50,  optimize_every = 1, random_seed = 42,  info = ''):
    thres_trial_av = 100
    #coeficient losses
    c_coding_av = .02 #rec neurons
    c_trials_av = 0.65 #rec av rates left
    c_time_av = 0.1666
    c_orth = 500
    c_sel = 0.5


    #loading data    
    mean_licks = mean_licks_inputs(files_list, hemi = hemi, max_time_index = max_time_index) 
    single_trial_licks = licks_inputs(files_list, max_time_index =  max_time_index)
    ALM_hemi= load_me_data(files_list, default_parameters, hemi=hemi, max_time_index=max_time_index)
    trial_average = tch.tensor(ALM_hemi['rates'][0:2,:,:], dtype = tch.float32).to(default_parameters['device']).float()
    print('dimensions', hemi, trial_average.shape, mean_licks.shape)
        
    #loading data    
    total_sessions = len(session_names)
    trial_indexes = {}
    for ses_name in session_names:
        trial_indexes[ses_name] = ALM_hemi['indexes'][ses_name]
    trial_average, trial_types = toch_version_ratates_trial_types(ALM_hemi, default_parameters, session_names)
    # single trial activity
    single_trials_activity = toch_single_trials(session_names, default_parameters, max_time_index = max_time_index)
    #indexes of neurons at each sesions
    indexes_neurons_st = [ALM_hemi['indexes'][s] for s in session_names]
    total_sessions = len(indexes_neurons_st )
    #parameters
    default_parameters['n_latents'] = n_pop
    default_parameters['n_neurons'] = ALM_hemi['rates'].shape[2]
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
    losses_coding_av = np.zeros(max_epochs)
    #Defining losses
    #average over time
    ListLossesTime = [LossAverageTime(default_parameters) for s in session_names]
    #average over neurons
    ListLossesCoding = [SelectivitySingleTrialLoss(trial_average, default_parameters, indexes_neurons_st[s]) for s in range(len(session_names))]

    #number of trials
    n_trials = 0
    for t_type in trial_types:
        n_trials+=len(trial_types[t_type])
    train_idx, test_idx = train_test_split(np.arange(n_trials), test_size=0.2, random_state=random_seed)
    default_parameters['n_trials'] = len(train_idx)
    net = SingleTrialLicks(default_parameters, indexes_neurons_st)
    net.device = default_parameters['device']
    if net_params is not None:
        try:
            net.load_state_dict(tch.load(net_params))
            print('Loading network')
        except:
            print('Different network sizes')
    optimizer = tch.optim.Adam(net.parameters(), lr=lr)
    epoch=0
    min_mean_loss = 50. 
    while epoch<max_epochs:
        #adding initial data
        # NOTE: single_trial_licks is a challenge for stitching, fix before stitch sessions
        net.t_start =  np.random.uniform(-4.35, -2.35)
        ind_t_start = int((-2.35-net.t_start)/net.dt)
        #train dataset
        zeros_insert = tch.zeros((2, len(train_idx), ind_t_start), device = 'cuda') 
        s_trials_licks = single_trial_licks[session_names[0]][:,train_idx,:]
        licks_with_zeros = tch.cat([zeros_insert, s_trials_licks], dim=2)
        #licks 
        zeros_insert = tch.zeros((2, ind_t_start, ALM_hemi['rates'].shape[2]), device = 'cuda') 
        mean_licks_zeros = tch.cat([zeros_insert, mean_licks], dim=1)
        t_types = trial_types[session_names[0]][train_idx]

        results, cov = net.forward(mean_licks_zeros, licks_with_zeros, t_types)
        #Reconsutring neural activity average over trials
        total_trial_average = LossTrials(trial_average, results['rates_alm'][:,ind_t_start:,:])
        losses_trial_av[epoch] = total_trial_average.detach().cpu().numpy()
    
        ##Reconsutring average activity over time
        total_time_average = 0
        for s in range(len(session_names)):
            #trial indexes to compute
            single_trial_neural_activity = single_trials_activity[session_names[s]][train_idx][:,:,indexes_neurons_st[s]]
            total_time_average += ListLossesTime[s](single_trial_neural_activity, results['rates_trials'][:,ind_t_start:,indexes_neurons_st[s]])
        total_time_average = total_time_average/float(total_sessions)
        losses_time_av[epoch] = total_time_average .detach().cpu().numpy()
    
        ##Reconsutring average activity over time
        total_coding_average = 0
        for s in range(len(session_names)):
            #trial indexes to compute
            single_trial_neural_activity = single_trials_activity[session_names[s]][train_idx][:,:,indexes_neurons_st[s]]
            total_coding_average += ListLossesCoding[s](net, single_trial_neural_activity, results['latents_trials'][:,ind_t_start:,:]) # Note: problem here with stitching
        total_coding_average  = total_coding_average /float(total_sessions)
        losses_coding_av[epoch] = total_coding_average.detach().cpu().numpy()

        #Loss orthogonality factors
        total_orth = LossWeights(cov)
        losses_weights[epoch] = total_orth.detach().cpu().numpy()

        #Loss selectivity
        total_selectivity = LossSelectivity(net)
        losses_sel[epoch] = total_selectivity.detach().cpu().numpy()

        # loss input trials
        l2_input_loss = tch.mean(net.input_trial ** 2)


        #defining total loss
        loss_factors =  c_orth * total_orth + c_sel * total_selectivity
        
        if thres_trial_av<losses_trial_av[epoch]:
            loss_reconstruction = total_trial_average
            net.is_trial_average = True
        else:
            net.is_trial_average = False
            loss_reconstruction = c_coding_av * total_coding_average + c_trials_av * total_trial_average + c_time_av * total_time_average
        total_loss = 0.7 * loss_reconstruction + 0.3 * loss_factors + 0.05 * l2_input_loss 
        total_loss= total_loss/ optimize_every

        #mean reconstruction loss
        mean_trial = np.mean(losses_trial_av[max(0, epoch-100):epoch])
        mean_total_loss = total_loss.detach().cpu().numpy()
        mean_time = np.mean(losses_time_av[max(0, epoch-100):epoch])
        mean_coding = np.mean(losses_coding_av[max(0, epoch-100):epoch])
        sel_loss = np.mean(losses_sel[max(0, epoch-100):epoch])
        weights_loss = np.mean(losses_weights[max(0, epoch-100):epoch])
        print(f'Epoch {epoch}'+f'| Total Loss {mean_total_loss}' +f' | Trial Av. = {mean_trial}'+f' | Epoch Av. = {mean_time}'+f' | Coding Av. = {mean_coding}'+ f'| Sel. = {sel_loss}'+f' | Orth. = {np.mean(weights_loss)}' +f' | Inputs = {l2_input_loss.detach().cpu().numpy()}' )
        
        ##optimizing
        optimizer.zero_grad()
        total_loss.backward()
        tch.nn.utils.clip_grad_norm_(net.parameters(),  max_norm=12., norm_type=2)
        optimizer.step()

            
        epoch += 1
        if epoch != 0 and epoch %test_every == 0 and mean_trial < min_mean_loss:
            min_mean_loss = min(min_mean_loss, mean_trial)
            print('saving net')
            name_f = path_save+'net_ALM_'+str(n_pop)+'_rseed_'+str(random_seed)+ '_train.pth'
            tch.save(net.state_dict(), name_f)   
        if epoch != 0 and epoch % test_every == 0:
            # Bundle everything into a single dict (include seed)
            losses_dict = {
                "meta": {
                    "epoch": epoch,
                    "n_pop": n_pop,
                    "test_every": test_every,
                    "random_seed": random_seed,
                },
                "trial_average": losses_trial_av,
                "time_average": losses_time_av,
                "projection": losses_coding_av,
                "weights": losses_weights,
                "selectivity": losses_sel,
            }

            os.makedirs(path_save, exist_ok=True)
            base = f"npop_{n_pop}_rseed_{random_seed}"
            save_path = os.path.join(path_save, f"losses_{base}_train.p")
            with open(save_path, "wb") as f:
                pickle.dump(losses_dict, f)

            # Plot (tag files with seed to avoid overwrites)
            plot_map = {
                "Loss Trial Average":  "trial_average",
                "Loss Epoch Average":  "time_average",
                "Projection Loss":     "projection",
            }
            for name_loss, key in plot_map.items():
                name_file = os.path.join(path_save, f"{key}_{base}")
                plot_loss(epoch, losses_dict[key], name_loss, tag=name_file)

 
