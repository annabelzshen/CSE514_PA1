import io
import warnings
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from os.path import exists
from contextlib import redirect_stdout

import pandas as pd
import warnings


# Plotting Support Functions
def configure_plots():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.set(style="ticks", color_codes=True, font_scale=1.5)
        sns.set_palette(sns.color_palette())
        plt.rcParams['figure.figsize'] = [16, 9]

        print('Plots configured! 📊')

        
def pair_plot(data, labels, names):
    n, d = data.shape
    hues = np.unique(labels)
    marks = itertools.cycle(('o', 's', '^', '.', 'd', ',')[:min(len(hues),6)])

    plt.figure(figsize=(16,12))
    _, axs = plt.subplots(d, d, sharex='col', sharey='row')

    for row in range(d):
        cat = data[:, row]

        # rescale y axes
        axs[row, 0].set_ylim(min(cat),max(cat))

        # set row and column labels
        axs[d-1, row].set_xlabel(names[row])
        axs[row, 0].set_ylabel(names[row])

        for column in range(d):
            ax = axs[row, column]

            # remove spines from top and right sides
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if column > row:
                ax.axis('off')
                continue

            if row == column:
                by_hue = [cat[np.where(labels == hue)] for hue in hues]
                ax.get_shared_y_axes().remove(ax)
                ax.autoscale()
                ax.hist(by_hue, bins=30, stacked=True)
            else:
                for hue in hues:
                    ax.scatter(data[:, column][np.where(labels == hue)],
                               data[:, row][np.where(labels == hue)],
                               20, marker=next(marks))

                    
def clean_columns(df):
    """Remove spaces and parentheses in column names."""
    df.columns = [
        col.replace(' ', '_').replace("(", "").replace(")", "").replace(
            ".", "") for col in df.columns
    ]
    return df


def plot_gender_fraction_over_time(df, title=None):
    """Calculates the annual fraction of artworks by gender. Plots the ratio over time."""

    # Drop entries where date is not defined
    df = df[df['DateAcquired'].notnull()]

    # Sort by date
    df = df.sort_values(by='DateAcquired')
    df = df.set_index('DateAcquired')

    # Add artworks acquired
    df['ones'] = 1
    df['num_acquired'] = df.ones.cumsum()
    

def configure_plots():
    '''Configures plots by making some quality of life adjustments'''
    for _ in range(2):
        plt.rcParams['figure.figsize'] = [16, 9]
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['lines.linewidth'] = 2
        
        
def optimize_logistic(X, y, theta=None, **kwargs):
    if theta is None:
        _, d = X.shape
        theta = np.zeros((d, 1))

    y = y.reshape(-1, 1)

    return optimize(logistic_gradient, X, y, theta, **kwargs).squeeze()
