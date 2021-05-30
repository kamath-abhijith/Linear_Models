'''

LOGISTIC REGRESSION ON TOY 10D NORMAL DATA

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.com

'''

# %% LOAD LIBRARIES

import os
import pickle
import argparse
import numpy as np

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils
import linear_tools

# # %% PARSE ARGUMENTS

parser = argparse.ArgumentParser(
    description = "LINEAR LEAST SQUARES CLASSIFICATION ON 10D TOY DATA"
)

parser.add_argument('--training_size', help="size of training set", type=int, default=10)
parser.add_argument('--force_train', help="force training", type=bool, default=True)

args = parser.parse_args()

training_size = args.training_size
force_train = args.force_train

# %% PLOT SETTINGS

# plt.style.use(['science','ieee'])

plt.rcParams.update({
    # "font.family": "serif",
    # "font.serif": ["cm"],
    # "mathtext.fontset": "cm",
    "font.size": 24})

# %% IMPORT DATA

train_data = np.loadtxt('./../data/Normal_train_10D.txt', delimiter=',', skiprows=1)
test_data = np.loadtxt('./../data/Normal_test_10D.txt', delimiter=',', skiprows=1)

num_train_samples, dim = train_data.shape

features = linear_tools.Polynomial_Features(1)

train_samples = features.transform(train_data[:,:dim-1])
train_labels = ((train_data[:, dim-1]+1)/2).astype(int)

test_samples = features.transform(test_data[:,:dim-1])
test_labels = ((test_data[:, dim-1]+1)/2).astype(int)

# %% TRAINING

os.makedirs('./../models/ex1', exist_ok=True)
path = './../models/ex1/'

num_samples, _ = train_samples.shape 
# training_size = num_samples
# force_train = False

if os.path.isfile(path + 'model_LOG_dataset_Normal' + '_size_' + \
    str(training_size) + '.pkl') and force_train==False:
    
    print('PICKING PRETRAINED MODEL')
    
    f = open(path + 'model_LOG_dataset_Normal' + '_size_' + \
        str(training_size) + '.pkl', 'rb')
    model = pickle.load(f)
    f.close()     

else:
    print('TRAINING IN PROCESS...')

    np.random.seed(34)
    random_idx = np.random.randint(num_samples, size=training_size)

    model = linear_tools.Logistic_Regression()
    model.fit(train_samples[random_idx], train_labels[random_idx])

    print('...TRAINING COMPLETE!')

    f = open(path + 'model_LOG_dataset_Normal' + '_size_' + str(training_size) + '.pkl', 'wb')
    pickle.dump(model, f)
    f.close()

# %% TESTING

# PREDICTIONS ON TEST DATA
confusion_mtx = model.accuracy(test_samples, test_labels)

# %% PLOT SAMPLES AND CONFUSION MATRIX

os.makedirs('./../results/ex1', exist_ok=True)
path = './../results/ex1/'

save_res = path + 'acc_LOG_dataset_Normal' + '_size_' + str(training_size)

utils.plot_confusion_matrix(confusion_mtx, show=False, save=save_res)
# %%
