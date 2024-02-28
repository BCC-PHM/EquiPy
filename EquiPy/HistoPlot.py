# HistoPlot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
    
matplotlib.rc ('font', **{'size'   : 16})
matplotlib.rc ('xtick', labelsize=14) 

def HistoPlot(
        data,
        index_col,
        count_col,
        metric_col,
        metric_label = "",
        index_label = "",
        index_perc_label = "Percentage with index value",
        ymax = None,
        color = "tab:purple"):

    # make sure top and right axes splines are there
    custom_params = {"axes.spines.right": True, "axes.spines.top": True}
    sns.set_theme(style="ticks", rc=custom_params)
    
    data.loc[:,"left"] = np.concatenate((np.array([0]), 
                                 np.cumsum(data[count_col].values)[:-1]))
    
    data.loc[:,"mid"] = data["left"].values + data[count_col].values/2
    
    fig = plt.figure(figsize = (8, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    ax1.bar(data["left"],
        data[metric_col],
        width = data[count_col], 
        align='edge', edgecolor = 'k', 
        linewidth = 2, color = color, alpha = 0.8)
    ax1.set_xticks(ticks = data["mid"], 
           labels = data[index_col],
           size = 14)
    
    ax1.set_ylim(0, ymax)
    ax1.set_xlim(0, sum(data[count_col]))
    ax1.set_ylabel(metric_label)
    ax1.set_xlabel(index_label)
    
    ax2.set_xlim(0, 100)
    ax2.set_xlabel(index_perc_label)

    return fig