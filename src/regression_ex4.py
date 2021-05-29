'''

LINEAR REGRESSION ON 1D DATA

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

data = np.loadtxt('./../data/1D_regression_data.txt', skiprows=1)

num_samples, dim = data.shape
order = 3

features = linear_tools.Polynomial_Features(order)
samples = features.transform(data[:,:dim-1])
labels = data[:,dim-1]

# %% TRAINING

model = linear_tools.Least_Squares_Regression()
model.fit(samples, labels)

# %% TESTING

test_samples = np.arange(-5,5,0.1)
test_features = features.transform(test_samples)
values = model.predict(test_features)

# %% PLOTS
