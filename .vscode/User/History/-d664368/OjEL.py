from training import *
import os
from os import listdir
from os.path import isfile, join
import pickle

#stitch all sessions
path_dandi= '../data/MAPDANDI_Functional/'
directory = os.fsencode(path_dandi)
files_list = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename[30:31]=='.':
        data = pickle.load(open(path_dandi+filename, 'rb'))
        if len(data)>0:
            n_neurons = data['average_trials'].shape[1]
            if n_neurons>0:
                files_list.append(filename)


n_pop = 12
net = None#'net_ALM_hemi_right8.pth'
hemi = 'right' #training on just left hemisphere neurons
max_epochs = 400000
def train(session_names = ['sub-456772_ses-20191120T115527.p'], n_pop=15, net_params=None, lr=5e-4, max_epochs = 500000, total_trials = 300, info = ''):

 
train_trial_average_licks(, net_params=net, n_pop=n_pop, hemi=hemi, lr=5e-4, max_epochs = max_epochs, c_av = 0.7, c_orth = 500, c_sel = 0.5, test_every = 50,  optimize_every = 1)







