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
        data,
        column, 
        value_pivot,
        eth_col = "Ethnicity", 
        IMD_col = "IMD",
        supp_thresh = 5):
    '''
    Parameters
    ----------
    data : DataFrame
        Source data containing ethnicity and deprivation columns
    value_pivot : DataFrame
        pivot table of counts accross
    supp_thresh : int, optional
        Values relating to counts below this value will be supressed.
        The default is 5.
    ttest: bool
        
    Returns
    -------
    values : numpy array
        Array of values that are allowed following supression
    labels : numpy array
        Array of labels.
    '''
    
    # Count number of cases associated with each value
    count_pivot = data.pivot_table(values = column, index = IMD_col, 
                                   columns = eth_col, aggfunc = "count")

    # Get pivot values and dimentions
    vals = value_pivot

    # supress labels
    labels = np.round(value_pivot,1).astype(str)
    labels[count_pivot < supp_thresh] = "Too\nsmall"
    
    # supress values
    vals[count_pivot < supp_thresh] = np.nan

    return vals, labels

def get_values(
        data, 
        column,
        eth_col = "Ethnicity", 
        IMD_col = "IMD",
        agg="mean"
        ):
    if agg=="mean":
        multiply = 100
        agg_col = "{} %".format(column)
    elif agg=="sum":
        agg_col = "Total number\n{}".format(column)
        multiply = 1
    else:
        agg_col = "agg val"
        multiply = 1
        
    data[agg_col] = multiply*data[column]
    eth_imd_piv = multiply * data.pivot_table(values = column, index = IMD_col, 
                                       columns = eth_col, aggfunc= agg)
    return eth_imd_piv, agg_col

def add_ttest(
        data,
        labels,
        column, 
        eth_col = "Ethnicity", 
        IMD_col = "IMD"):
    '''
    Parameters
    ----------
    data : DataFrame
        Source data containing ethnicity and deprivation columns
    labels : DataFrame
        Plot value labels.
    column : string
        Column for .
    eth_col : string, optional
        Column header for ethnicity column. The default is "Ethnicity".
    IMD_col : string, optional
        Column header for deprivation index column. The default is "IMD".
        
    Returns
    -------
    labels : DataFrame
        Two-sample t-test on number of samples.

    '''
    # https://medium.com/analytics-vidhya/testing-a-difference-in-population-proportions-in-python-89d57a06254
    
    # Get sample means
    x1 = data.pivot_table(values = column, index = IMD_col, 
                          columns = eth_col, aggfunc= "sum")

    
    # Get number in each sample
    n1 = data.pivot_table(values = column, index = IMD_col, 
                          columns = eth_col, aggfunc= "count")
    
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
def inequality_map(data, 
                   column, 
                   palette = "Purples",
                   eth_col = "Ethnicity", 
                   IMD_col = "IMD",
                   agg="mean",
                   letter = "",
                   supp_thresh = 5,
                   ttest = False):
    
    # Pivot data to get plot raw values
    plot_vals, agg_col = get_values(
        data, 
        column, 
        eth_col, 
        IMD_col,
        agg
        )
    
    # Apply small number supression
    plot_vals, labels = small_number_supression(
            data,
            column, 
            plot_vals,
            eth_col = eth_col, 
            IMD_col = IMD_col,
            supp_thresh = supp_thresh)

    if ttest:
        labels = add_ttest(
            data,
            labels,
            column, 
            eth_col = eth_col, 
            IMD_col = IMD_col
            )

    # Get bar color that matches the chosen palette
    bar_col = plt.colormaps[palette](0.8)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(8, 8)

    ax1 = fig.add_subplot(gs[2:8, :6])
    sns.heatmap(plot_vals, annot=labels, fmt="",
                linewidths=.5, ax=ax1, cmap = palette, cbar=False)

    ax1.set_yticklabels(ax1.get_yticks(), rotation = 0)
    
    # TODO: Develop dynamic solution for this labelling
    ax1.set_yticklabels(["1\nMost\ndeprived","2","3","4","5\nLeast\ndeprived"])
    
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 0)
    ax1.set_xticklabels(np.unique(data[eth_col]))
    ax1.set_ylabel("IMD Quintile")
    
    ax2 = fig.add_subplot(gs[:2, :6])
    bar1 = sns.barplot(data, x = eth_col, y = agg_col,
                  color = bar_col,
                  order = np.unique(data[eth_col]),
                  estimator = agg)
    bar1.get_lines()[0].get_data()
    ax2.set_xticks([])
    ax2.set_xlabel("")

    ax3 = fig.add_subplot(gs[2:, 6:])
    sns.barplot(data, y = IMD_col,
                x = agg_col,
                color = bar_col,
                order = [1,2,3,4,5],orient="h",
                estimator = agg
                )

    ax3.set_yticks([])
    ax3.set_ylabel("")
    
    ax1.annotate(letter, (0.82, 0.82), xycoords='figure fraction',
                 size = 22)
    
    return fig

