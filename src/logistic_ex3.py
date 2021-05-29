'''

LOGISTIC REGRESSION ON GERMAN NUMERIC

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

data = np.loadtxt('./../data/german.data-numeric', skiprows=1)

num_samples, dim = data.shape

features = linear_tools.Polynomial_Features(1)

samples = features.normalise(data[:,:dim-1])
samples = features.transform(samples)
labels = (data[:,dim-1] - 1).astype(int)
num_classes = len(np.unique(labels))

dataset = linear_tools.Dataset(samples, labels)

# %% TRAINING AND TESTING

iter_len = 100
confusion_mtx = np.zeros((iter_len, num_classes, num_classes))
for iter in range(iter_len):
    train_samples, train_labels, test_samples, test_labels = \
        dataset.train_test_split(samples, labels, fraction=0.5)

    # TRAINING

    model = linear_tools.Logistic_Regression()
    model.fit(train_samples, train_labels)

    # TESTING

    confusion_mtx[iter, :, :] = model.accuracy(test_samples, test_labels)

confusion_mtx = np.mean(confusion_mtx, axis=0)
LOG_ACCURACY = sum(np.diag(confusion_mtx))/sum(sum(confusion_mtx)) * 100
print(r'Logistic Regression classifies with %.2f %% accuracy' %(LOG_ACCURACY))

# %%
utils.plot_confusion_matrix(confusion_mtx)
# %%