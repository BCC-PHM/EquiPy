# -*- coding: utf-8 -*-
"""
Inequality matrix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    vals[count_pivot < supp_thresh] = np.nan

    return vals, labels

# define inequality matrix function
def inequality_map(data, 
                   column, 
                   palette = "Purples",
                   eth_col = "Ethnicity", 
                   IMD_col = "IMD",
                   agg="mean",
                   letter = "",
                   supp_thresh = 5):
    
    # Get bar color that matches the chosen palette
    bar_col = plt.colormaps[palette](0.8)
    
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
    eth_imd_piv = data.pivot_table(values = column, index = IMD_col, 
                                       columns = eth_col, aggfunc= agg)
    
    # Apply small number supression
    vals, labels = small_number_supression(
            data,
            column, 
            multiply*eth_imd_piv,
            eth_col = eth_col, 
            IMD_col = IMD_col,
            supp_thresh = supp_thresh)

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(8, 8)

    ax1 = fig.add_subplot(gs[2:8, :6])
    sns.heatmap(vals, annot=labels, fmt="",
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

