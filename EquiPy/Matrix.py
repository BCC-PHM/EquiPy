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


def small_number_suppression(
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
        Array of values that are allowed following suppression
    labels : numpy array
        Array of labels.
    '''
    
    # Get pivot values and dimentions
    supressed_pivot = plot_pivot
    supressed_pivot[count_pivot < supp_thresh] = 0 
    labels = np.round(plot_pivot,1).astype(str)
    
    # Add percentage symbol if needed 
    if any((plot_pivot.values != count_pivot.values).flatten()):
        labels = labels + "%"

    # supress labels
    labels[count_pivot < supp_thresh] = "Too\nsmall"
    labels[count_pivot.isnull()] = "0"

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
        output_pivot[output_pivot.isnull()] = 0
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
    
    # Get number in each sample
    n1 = count_pivot 
    p1 = plot_pivot/100
    x1 = n1*p1
    
    # Get reference values
    # TODO: Make it so that a different cell can be set a reference
    n2 = n1.values[-1,-1]
    p2 = p1.values[-1,-1]
    x2 = n2*p2
    
    p_star = (x1 + x2) / (n1 + n2)
    variance = p_star * (1 - p_star)

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
                   title = "",
                   letter = "",
                   supp_thresh = 5,
                   ttest = False,
                   IMD_ticks = ["1\nMost\ndeprived","2","3","4","5\nLeast\ndeprived"],
                   CI_method = None,
                   width = 7,
                   height = 6,
                   Z = 1.96
                   ):
    
    # If no percentage pivot given, just plot the count
    if type(perc_pivot) != pd.core.frame.DataFrame:
        plot_pivot = count_pivot
        bar_x = np.sum(count_pivot, axis = 0)
        bar_y = np.sum(count_pivot, axis = 1)
    else:
        plot_pivot = perc_pivot
        bar_x = np.sum(count_pivot*perc_pivot, axis = 0) / np.sum(count_pivot, axis = 0)
        bar_y = np.sum(count_pivot*perc_pivot, axis = 1) / np.sum(count_pivot, axis = 1)

    # Apply small number suppression
    supressed_pivot, labels = small_number_suppression(
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
    bar_col = plt.colormaps[palette](0.7)

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(8, 8)

    ax1 = fig.add_subplot(gs[2:8, :6])
    sns.heatmap(supressed_pivot, annot=labels, fmt="",
                linewidths=.5, ax=ax1, cmap = palette, cbar=False)

    ax1.set_yticklabels(ax1.get_yticks(), rotation = 0)
    
    # TODO: Develop dynamic solution for this labelling
    ax1.set_yticklabels(IMD_ticks)
    
    ax1.set_xticklabels(ax1.get_xticks(), rotation = 0)
    ax1.set_xticklabels(plot_pivot.columns)
    ax1.set_ylabel("IMD Quintile")
    
    ax2 = fig.add_subplot(gs[:2, :6])
    bar1 = sns.barplot(x = bar_x.index, y = bar_x,  color = bar_col)
    #sns.barplot(plot_pivot, color = bar_col)
    bar1.get_lines()[0].get_data()
    ax2.set_xticks([])
    ax2.set_xlabel("")
    ax2.set_ylabel(title)

    ax3 = fig.add_subplot(gs[2:, 6:])
    sns.barplot(x = bar_y, y = bar_y.index, color = bar_col, orient = "h")

    ax3.set_yticks([])
    ax3.set_ylabel("")
    ax3.set_xlabel(title)
    
    # Calculate uncertainties
    if (type(CI_method) == str) and ( type(perc_pivot) == pd.core.frame.DataFrame):
        yerror = calc_CI(
            count_pivot, 
            perc_pivot,
            axis = 0,
            CI_method = CI_method,
            Z = Z
            )

        # top bar plot
        ax2.errorbar(
            x = ax1.get_xticks() - 0.5, 
            y = bar_x,
            yerr = yerror,
            fmt ='o',
            color = "k",
            ms = 0)
        
        
        xerror = calc_CI(
            count_pivot, 
            perc_pivot,
            axis = 1,
            CI_method = CI_method,
            Z = Z
            )
        # Right hand bar plot
        ax3.errorbar(
            x = bar_y, 
            y = ax1.get_yticks()- 0.5,
            xerr = xerror,
            fmt ='o',
            color = "k",
            ms = 0)
    
        
    # Fix barplot axis so that they match
    perc_max = max(ax2.get_ylim()[1], ax1.get_xlim()[1])
    bar_max = int(np.ceil(perc_max / 5.0)) * 5
    ax2.set_ylim(0, bar_max)
    ax3.set_xlim(0, bar_max)
    
    ax1.annotate(letter, (0.82, 0.82), xycoords='figure fraction',
                 size = 22)
    
    return fig

def calc_CI(count_pivot, 
            perc_pivot,
            axis = 0,
            CI_method = "Wilson",
            Z = 1.96):
            
    assert CI_method in ["Wilson"],  "CI_method not recognised."
    
    n = np.sum(perc_pivot * count_pivot / 100, axis = axis)
    N = np.sum(count_pivot, axis = axis)
    
    p_hat = n/N
    
    CI_lower = 100 * (p_hat + Z**2/(2*N) - Z * np.sqrt((p_hat*(1-p_hat)/N) + Z**2/(4*N**2))) / (1 + Z**2/N)
    CI_upper = 100 * (p_hat + Z**2/(2*N) + Z * np.sqrt((p_hat*(1-p_hat)/N) + Z**2/(4*N**2))) / (1 + Z**2/N)
    
    
    return [100*p_hat - CI_lower, CI_upper- 100*p_hat]
    
    