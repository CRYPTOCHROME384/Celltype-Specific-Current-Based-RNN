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

files_list = ['sub-442571_ses-20190228T140832.p']
n_pop = 10
net_params = None#'net_ALM_20_5.pth'
hemi = 'right' #training on just left hemisphere neurons
max_epochs = 400000
max_time_index = 275
random_seed = 38
train(files_list, session_names = ['sub-442571_ses-20190228T140832.p'], net_params=net_params, n_pop=n_pop, hemi=hemi, max_time_index =max_time_index,  lr=5e-4, max_epochs = max_epochs,  test_every = 50,  optimize_every = 1, random_seed=random_seed)

 






