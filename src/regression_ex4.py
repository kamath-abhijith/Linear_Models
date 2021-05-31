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

from tqdm import tqdm

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils
import linear_tools

# %% PARSE ARGUMENTS

parser = argparse.ArgumentParser(
    description = "LINEAR REGRESSION ON 1D DATA"
)

parser.add_argument('--training_size', help="Size of training set", type=int, default=50)
parser.add_argument('--subsample', help="Method of subsampling", default='uniform')

args = parser.parse_args()

training_size = args.training_size
subsample = args.subsample

# %% PLOT SETTINGS

# plt.style.use(['science','ieee'])

plt.rcParams.update({
    # "font.family": "serif",
    # "font.serif": ["cm"],
    # "mathtext.fontset": "cm",
    "font.size": 24})

# %% IMPORT DATA

data = np.loadtxt('./../data/1D_regression_data.txt', skiprows=1)

num_samples, dim = data.shape
# subsample = 'uniform'
# training_size = 50

if subsample == 'random':
    random_idx = np.random.randint(num_samples, size=training_size)
    data = data[random_idx]
elif subsample == 'uniform':
    uniform_idx = np.linspace(0, num_samples-1, training_size, dtype=int)
    data = data[uniform_idx]

features_order2 = linear_tools.Polynomial_Features(2)
samples_order2 = features_order2.transform(data[:,:dim-1])

features_order3 = linear_tools.Polynomial_Features(3)
samples_order3 = features_order3.transform(data[:,:dim-1])

features_order4 = linear_tools.Polynomial_Features(4)
samples_order4 = features_order4.transform(data[:,:dim-1])

labels = data[:,dim-1]

# %% TRAINING AND TESTING

num_iter = 100
error_order2 = np.zeros(num_iter)
error_order3 = np.zeros(num_iter)
error_order4 = np.zeros(num_iter)
for itr in tqdm(range(num_iter)):
    model_order2 = linear_tools.Least_Squares_Regression()
    model_order2.fit(samples_order2, labels)

    model_order3 = linear_tools.Least_Squares_Regression()
    model_order3.fit(samples_order3, labels)

    model_order4 = linear_tools.Least_Squares_Regression()
    model_order4.fit(samples_order4, labels)

    test_samples = np.arange(-5,5,0.1)
    true_weights = np.array([-3.0, -3.0, 1.25, 0.25])

    test_features_order2 = features_order2.transform(test_samples)
    values_order2 = model_order2.predict(test_features_order2)

    test_features_order3 = features_order3.transform(test_samples)
    values_order3 = model_order3.predict(test_features_order3)
    true_signal = test_features_order3 @ true_weights

    test_features_order4 = features_order4.transform(test_samples)
    values_order4 = model_order4.predict(test_features_order4)

    error_order2[itr] = np.linalg.norm(true_signal - values_order2)
    error_order3[itr] = np.linalg.norm(true_signal - values_order3)
    error_order4[itr] = np.linalg.norm(true_signal - values_order4)

error_order2_mean = np.mean(error_order2)
error_order3_mean = np.mean(error_order3)
error_order4_mean = np.mean(error_order4)

# %% PLOTS

os.makedirs('./../results/ex4', exist_ok=True)
path = './../results/ex4/'
save_res = path + 'samples_LS_regression_subsample_' + subsample + '_size_' + str(training_size)

plt.figure(figsize=(8,8))
ax = plt.gca()

utils.plot_samples1D(data[:,:dim-1], labels, ax=ax, plot_colour='blue',
    legend_label=r'Samples', show=False)
utils.plot_signal(test_samples, values_order2, ax=ax, plot_colour='red',
    ylimits=[-10,25], legend_label=r'Degree $2$, %.2f' %(error_order2_mean),
    line_width=4, line_style='-', show=False)
utils.plot_signal(test_samples, values_order3, ax=ax, plot_colour='green',
    ylimits=[-10,25], legend_label=r'Degree $3$, %.2f' %(error_order3_mean),
    line_width=4, line_style='-', show=False)
utils.plot_signal(test_samples, values_order4, ax=ax, plot_colour='magenta',
    ylimits=[-10,25], legend_label=r'Degree $4$, %.2f' %(error_order4_mean),
    line_width=4, line_style='-', show=False)
utils.plot_signal(test_samples, true_signal, ax=ax, plot_colour='blue',
    ylimits=[-10,25], legend_label=r'True Polynomial', line_width=4,
    line_style='--', show=False, save=save_res)
# %%
