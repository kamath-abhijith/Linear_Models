'''

LINEAR LEAST-SQUARES CLASSIFICATION ON IRIS

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
samples = data[:, :dim-1]
labels = (data[:, dim-1] - 1).astype(int)

# %% TRAINING

model = linear_tools.Least_Squares_Classifier()
model.fit(samples, labels)

# %% TESTING

confusion_mtx = model.accuracy(samples, labels)

# %%
utils.plot_confusion_matrix(confusion_mtx)