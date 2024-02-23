# -*- coding: utf-8 -*-
"""
Inequality matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def small_number_supression(
        count_pivot,
        plot_pivot, 
        supp_thresh = 5
        ):
    '''
    Parameters
    ----------
    count_pivot : DataFrame
        
    plot_pivot : DataFrame
        
    supp_thresh : int, optional
        Values relating to counts below this value will be supressed.
        The default is 5.
        
    Returns
    -------
    supressed_pivot : numpy array
        Array of values that are allowed following supression
    labels : numpy array
        Array of labels.
    '''
    
    # Get pivot values and dimentions
    supressed_pivot = plot_pivot

    # supress labels
    labels = np.round(plot_pivot,1).astype(str)
    labels[count_pivot < supp_thresh] = "Too\nsmall"
    
    # supress values
    supressed_pivot[count_pivot < supp_thresh] = np.nan

    return supressed_pivot, labels

def get_pivot(
        data, 
        column = None,
        eth_col = "Ethnicity", 
        IMD_col = "IMD",
        mode="percentage"
        ):
    
    if mode=="percentage":
        
        if column == None:
            assert("Undefined variable column.")
        
        output_pivot = 100 * data.pivot_table(values = column, index = IMD_col, 
                                              columns = eth_col, aggfunc = "mean")
    elif mode=="count":
        data["index"] = range(len(data))
        output_pivot = data.pivot_table(values = "index", index = IMD_col,
                                        columns = eth_col, aggfunc = "count")
    else:
        assert("Mode not recognised. Please set mode = 'percentage' or 'count'.")
        
    return output_pivot

def add_ttest(
    count_pivot,
    plot_pivot,
    labels
    ):
    '''
    Parameters
    ----------
    count_pivot : DataFrame
        
    plot_pivot : DataFrame
        
    labels : numpy array
        Array of labels.
        
    Returns
    -------
    labels : DataFrame
        Two-sample t-test on number of samples.
    '''
    # https://medium.com/analytics-vidhya/testing-a-difference-in-population-proportions-in-python-89d57a06254
    
    # Get sample means
    x1 = count_pivot

    # Get number in each sample
    n1 = plot_pivot/100
    
    p1 = x1/n1
    
    # Get reference values
    # TODO: Make it so that a different cell can be set a reference
    x2 = x1.values[-1,-1]
    n2 = n1.values[-1,-1]
    
    p2 = x2/n2
    
    p_star = (x1 + x2) / (n1 + n2)
    variance = p_star *(1 - p_star)
    standard_error = np.sqrt( variance * (1 / n1 + 1 / n2) ) 
    z_star = (p1 - p2) / standard_error
    
    p_score = 2*norm.cdf(-np.abs(z_star))
    
    sigs = np.full(labels.shape, "", dtype=object)
    
    sigs[p_score <= 0.1 ] = "*"
    sigs[p_score <= 0.05 ] = "**"
    sigs[p_score <= 0.001 ] = "***"
    

    labels = labels + "\n" + sigs
    labels.iloc[-1,-1] = labels.iloc[-1,-1] + "(Ref)"


    return labels


# define inequality matrix function
def inequality_map(count_pivot, 
                   perc_pivot = None, 
                   palette = "Purples",
                   letter = "",
                   supp_thresh = 5,
                   ttest = False):
    
    if type(perc_pivot) != pd.core.frame.DataFrame:
        plot_pivot = count_pivot
    else:
        plot_pivot = perc_pivot


    # Apply small number supression
    supressed_pivot, labels = small_number_supression(
            count_pivot,
            plot_pivot, 
            supp_thresh = supp_thresh)
    

    if ttest and ( type(perc_pivot) == pd.core.frame.DataFrame):
        labels = add_ttest(
            count_pivot,
            plot_pivot,
            labels
            )

    # Get bar color that matches the chosen palette
    bar_col = plt.colormaps[palette](0.8)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(8, 8)

    ax1 = fig.add_subplot(gs[2:8, :6])
    sns.heatmap(plot_pivot, annot=labels, fmt="",
                linewidths=.5, ax=ax1, cmap = palette, cbar=False)

    ax1.set_yticklabels(ax1.get_yticks(), rotation = 0)
    
    # TODO: Develop dynamic solution for this labelling
    ax1.set_yticklabels(["1\nMost\ndeprived","2","3","4","5\nLeast\ndeprived"])
    
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 0)
    ax1.set_xticklabels(perc_pivot.columns)
    ax1.set_ylabel("IMD Quintile")
    
    ax2 = fig.add_subplot(gs[:2, :6])
    bar1 = sns.barplot(plot_pivot, color = bar_col)
    sns.barplot(plot_pivot, color = bar_col)
    bar1.get_lines()[0].get_data()
    ax2.set_xticks([])
    ax2.set_xlabel("")

    ax3 = fig.add_subplot(gs[2:, 6:])
    sns.barplot(plot_pivot.T, color = bar_col)

    ax3.set_yticks([])
    ax3.set_ylabel("")
    
    ax1.annotate(letter, (0.82, 0.82), xycoords='figure fraction',
                 size = 22)
    
    return fig

