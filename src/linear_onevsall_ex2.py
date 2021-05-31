'''

LINEAR LEAST-SQUARES CLASSIFICATION ON IRIS, ONE VS. ALL

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.com

'''

# %% LOAD LIBRARIES

import os
import pickle
import argparse
import numpy as np

from tqdm import tqdm

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils
import linear_tools

# %% PARSE ARGUMENTS

parser = argparse.ArgumentParser(
    description = "LOGISTIC REGRESSION ON IRIS, MULTICLASS"
)

parser.add_argument('--split_fraction', help="fractions for train and test", type=float, default=0.5)
# parser.add_argument('--force_train', help="force training", type=bool, default=True)

args = parser.parse_args()

split_fraction = args.split_fraction
# force_train = args.force_train

# %% PLOT SETTINGS

# plt.style.use(['science','ieee'])

plt.rcParams.update({
    # "font.family": "serif",
    # "font.serif": ["cm"],
    # "mathtext.fontset": "cm",
    "font.size": 24})

# %% IMPORT DATA

data = np.loadtxt('./../data/iris_dataset.txt', delimiter=',', skiprows=1)

num_samples, dim = data.shape

features = linear_tools.Polynomial_Features(1)

samples = features.transform(data[:,:dim-1])
labels = (data[:,dim-1] - 1).astype(int)
num_classes = len(np.unique(labels))

# split_fraction = 0.5

# %% TRAINING AND TESTING, CLASS 0

class0_labels = labels.copy()
class0_labels[np.where(class0_labels==2)] = 1

dataset = linear_tools.Dataset(samples, class0_labels)

iter_len = 10000
confusion_mtx = np.zeros((iter_len, 2, 2))
for iter in tqdm(range(iter_len)):
    train_samples, train_labels, test_samples, test_labels = \
        dataset.train_test_split(samples, class0_labels, fraction=split_fraction)

    # TRAINING

    model = linear_tools.Least_Squares_Classifier()
    model.fit(train_samples, train_labels)

    # TESTING

    confusion_mtx[iter, :, :] = model.accuracy(test_samples, test_labels,
        return_type='absolute')

confusion_mtx_class0 = np.mean(confusion_mtx, axis=0)
ACCURACY_0 = sum(np.diag(confusion_mtx_class0))/sum(sum(confusion_mtx_class0))*100

# %% TRAINING AND TESTING, CLASS 1

class1_labels = labels.copy()
class1_labels[np.where(class1_labels==2)] = 0
class1_labels = np.invert(class1_labels) + 2

dataset = linear_tools.Dataset(samples, class1_labels)

iter_len = 10000
confusion_mtx = np.zeros((iter_len, 2, 2))
for iter in tqdm(range(iter_len)):
    train_samples, train_labels, test_samples, test_labels = \
        dataset.train_test_split(samples, class1_labels, fraction=split_fraction)

    # TRAINING

    model = linear_tools.Least_Squares_Classifier()
    model.fit(train_samples, train_labels)

    # TESTING

    confusion_mtx[iter, :, :] = model.accuracy(test_samples, test_labels,
        return_type='absolute')

confusion_mtx_class1 = np.mean(confusion_mtx, axis=0)
ACCURACY_1 = sum(np.diag(confusion_mtx_class1))/sum(sum(confusion_mtx_class1))*100

# %% TRAINING AND TESTING, CLASS 2

class2_labels = labels.copy()
class2_labels[np.where(class2_labels==1)] = 0
class2_labels = class2_labels/2
class2_labels = np.invert(class2_labels.astype(int)) + 2

dataset = linear_tools.Dataset(samples, class2_labels)

iter_len = 10000
confusion_mtx = np.zeros((iter_len, 2, 2))
for iter in tqdm(range(iter_len)):
    train_samples, train_labels, test_samples, test_labels = \
        dataset.train_test_split(samples, class2_labels, fraction=split_fraction)

    # TRAINING

    model = linear_tools.Least_Squares_Classifier()
    model.fit(train_samples, train_labels)

    # TESTING

    confusion_mtx[iter, :, :] = model.accuracy(test_samples, test_labels,
        return_type='absolute')

confusion_mtx_class2 = np.mean(confusion_mtx, axis=0)
ACCURACY_2 = sum(np.diag(confusion_mtx_class2))/sum(sum(confusion_mtx_class2))*100

# %% PLOTS

os.makedirs('./../results/ex2', exist_ok=True)
path = './../results/ex2/'

save_res = path + 'acc_LSOA_dataset_iris' + '_fraction_' + str(split_fraction)

utils.plot_confusion_matrix(confusion_mtx_class0, map_min=None, map_max=None,
    title_text=r'ACCURACY: %.2f %%' %(ACCURACY_0), show=False, save=save_res+'_class_0')
utils.plot_confusion_matrix(confusion_mtx_class1, map_min=None, map_max=None,
    title_text=r'ACCURACY: %.2f %%' %(ACCURACY_1), show=False, save=save_res+'_class_1')
utils.plot_confusion_matrix(confusion_mtx_class2, map_min=None, map_max=None,
    title_text=r'ACCURACY: %.2f %%' %(ACCURACY_2), show=False, save=save_res+'_class_2')
# %%
