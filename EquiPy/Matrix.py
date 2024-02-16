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

# define inequality matrix function
def inequality_map(data, 
                   column, 
                   palette = "Purples",
                   eth_col = "Ethnicity Group", 
                   IMD_col = "IMD Quintile - ALL",
                   agg="mean",
                   fmt=".1f",
                   letter = ""):
    
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

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(8, 8)

    ax1 = fig.add_subplot(gs[2:8, :6])
    sns.heatmap(multiply*eth_imd_piv, annot=True, fmt = fmt, linewidths=.5, 
                ax=ax1, cmap = palette, cbar=False)

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

