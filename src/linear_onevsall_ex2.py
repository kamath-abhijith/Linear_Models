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

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils
import linear_tools

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% IMPORT DATA

data = np.loadtxt('./../data/iris_dataset.txt', delimiter=',', skiprows=1)

num_samples, dim = data.shape

features = linear_tools.Polynomial_Features(1)

samples = features.transform(data[:,:dim-1])
labels = (data[:,dim-1] - 1).astype(int)
num_classes = len(np.unique(labels))

# %% TRAINING AND TESTING, CLASS 0

class0_labels = labels.copy()
class0_labels[np.where(class0_labels==2)] = 1

dataset = linear_tools.Dataset(samples, class0_labels)

iter_len = 100
confusion_mtx = np.zeros((iter_len, 2, 2))
for iter in range(iter_len):
    train_samples, train_labels, test_samples, test_labels = \
        dataset.train_test_split(samples, class0_labels, fraction=0.5)

    # TRAINING

    model = linear_tools.Least_Squares_Classifier()
    model.fit(train_samples, train_labels)

    # TESTING

    confusion_mtx[iter, :, :] = model.accuracy(test_samples, test_labels)

confusion_mtx_class0 = np.mean(confusion_mtx, axis=0)

# %% TRAINING AND TESTING, CLASS 1

class1_labels = labels.copy()
class1_labels[np.where(class1_labels==2)] = 0
class1_labels = np.invert(class1_labels) + 2

dataset = linear_tools.Dataset(samples, class1_labels)

iter_len = 100
confusion_mtx = np.zeros((iter_len, 2, 2))
for iter in range(iter_len):
    train_samples, train_labels, test_samples, test_labels = \
        dataset.train_test_split(samples, class1_labels, fraction=0.5)

    # TRAINING

    model = linear_tools.Least_Squares_Classifier()
    model.fit(train_samples, train_labels)

    # TESTING

    confusion_mtx[iter, :, :] = model.accuracy(test_samples, test_labels)

confusion_mtx_class1 = np.mean(confusion_mtx, axis=0)

# %% TRAINING AND TESTING, CLASS 2

class2_labels = labels.copy()
class2_labels[np.where(class2_labels==1)] = 0
class2_labels = class2_labels/2
class2_labels = np.invert(class2_labels.astype(int)) + 2

dataset = linear_tools.Dataset(samples, class2_labels)

iter_len = 100
confusion_mtx = np.zeros((iter_len, 2, 2))
for iter in range(iter_len):
    train_samples, train_labels, test_samples, test_labels = \
        dataset.train_test_split(samples, class2_labels, fraction=0.5)

    # TRAINING

    model = linear_tools.Least_Squares_Classifier()
    model.fit(train_samples, train_labels)

    # TESTING

    confusion_mtx[iter, :, :] = model.accuracy(test_samples, test_labels)

confusion_mtx_class2 = np.mean(confusion_mtx, axis=0)

# %% PLOTS

utils.plot_confusion_matrix(confusion_mtx_class0)
utils.plot_confusion_matrix(confusion_mtx_class1)
utils.plot_confusion_matrix(confusion_mtx_class2)