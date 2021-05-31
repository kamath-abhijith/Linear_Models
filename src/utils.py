'''

CONTAINS UTILITY FUNCTIONS FOR LINEAR_TOOLS

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

import linear_tools

# %% PLOTTING

def plot_data2D(data, ax=None, title_text=None,
    xlimits=[-4,10], ylimits=[-4,10], show=True, save=False):
    ''' Plots 2D data with labels '''
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    labels = data[:,2]
    pos_samples = data[np.where(labels==1)][:,:2]
    neg_samples = data[np.where(labels==-1)][:,:2]

    plt.scatter(pos_samples[:,0], pos_samples[:,1], color='red')
    plt.scatter(neg_samples[:,0], neg_samples[:,1], color='green')

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title_text)

    plt.ylim(ylimits)
    plt.xlim(xlimits)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_decisionboundary2D(x1, x2, labels, ax=None,
    xlimits=[-1,20], ylimits=[-1,20], show=True, save=False):
    '''
    Plots the decision boundary for linear classifier

    '''
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    plt.contourf(x2, x2, labels, alpha=0.2, levels=np.linspace(0, 1, 3))
    plt.xlim(xlimits)
    plt.ylim(ylimits)

    if save:
        plt.savefig(save + '.pdf', format='pdf')
    
    if show:
        plt.show()

    return

def plot_confusion_matrix(data, ax=None, xaxis_label=r'PREDICTED CLASS',
    yaxis_label=r'TRUE CLASS', map_min=0.0, map_max=1.0, title_text=None,
    show=True, save=False):
    ''' Plots confusion matrix '''
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    ax = sns.heatmap(data, vmin=map_min, vmax=map_max, linewidths=0.5,
        annot=True)
    # ax.invert_yaxis()
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_samples1D(x, y, ax=None, title_text=None, plot_colour='blue',
    xlimits=[-5,5], ylimits=[0,5], legend_label=None, legend_show=True,
    legend_loc='upper left', show=True, save=False):
    ''' Plots 1D samples for regression '''

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    plt.scatter(x, y, color=plot_colour, label=legend_label)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(title_text)

    plt.ylim(ylimits)
    plt.xlim(xlimits)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_signal(x, y, ax=None, title_text=None, plot_colour='blue',
    xlimits=[-6,6], ylimits=[-10,25], legend_label=None, legend_show=True,
    legend_loc='upper left', line_style='-', line_width=None,
    show=True, save=False):
    ''' Plots 1D samples for regression '''

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    plt.plot(x, y, color=plot_colour, linestyle=line_style,
        linewidth=line_width, label=legend_label)

    if legend_label and legend_show:
        plt.legend(ncol=1, loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(title_text)

    plt.ylim(ylimits)
    plt.xlim(xlimits)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return