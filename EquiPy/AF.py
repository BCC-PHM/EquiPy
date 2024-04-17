# -*- coding: utf-8 -*-
"""
Attributable fraction for the exposed group

AF = (RR-1)/RR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def outcome_count(df, observable, ref_IMD = "3+", 
                  ref_eth = "White",
                  eth_col = "Ethnicity Group",
                  IMD_col = "IMD Quintile",
                  scale_factor = 1):
    count = df.groupby([eth_col, IMD_col])[observable].agg(["sum"]).reset_index()
    count_piv = pd.pivot_table(count, values = "sum", index = IMD_col, 
                               columns = eth_col)
    
    # Recale values (Used for datasets inflatted by imputation)
    if scale_factor != 1:
        count_piv = np.round(count_piv/scale_factor).astype(int)
    return count_piv

def get_AF(df, observable, ref_IMD = "3+", 
           ref_eth = "White",
           mode = "AF",
           eth_col = "Ethnicity Group",
           IMD_col = "IMD Quintile"):
    '''
    inputs:
        mode - Attributable proportion (AF) or population attributable
        proportion (PAF)
    '''
    #print(eth_col, IMD_col)
    risk =  df.groupby([eth_col, IMD_col])[observable].agg(["mean", "count"]).reset_index()
    #print(risk)
    ref_mask = np.logical_and(risk[IMD_col] == ref_IMD,
                              risk[eth_col] == ref_eth)
    
    # risk for unexposed group
    R_u = risk["mean"][ref_mask].values[0]
    
    risk["RR"] = risk["mean"] / R_u
    risk["Pe"] = risk["count"]/len(df)
    
    if mode == "PAF":
        risk["AF_p"] = risk["Pe"] * (risk["RR"] - 1) / (1 + risk["Pe"] * (risk["RR"] - 1))
    elif mode == "AF":
        risk["AF_p"] = (risk["RR"] - 1) / risk["RR"]
    else:
        raise "Unexpected AF mode"
        
    risk["AF_p %"] = np.round(100*risk["AF_p"], 1)
    
    pivot = pd.pivot_table(risk, values = "AF_p %", index = IMD_col, columns = eth_col)
    
    return pivot


def new_bootstrap_AF(df, observable, 
                     ref_IMD = "3+", 
                     ref_eth = "White",
                     eth_col = "Ethnicity Group",
                     IMD_col = "IMD Quintile"):
    df = df.sample(frac=1, replace=True)
    pivot = get_AF(df, observable, ref_IMD, ref_eth,
                   eth_col = eth_col, IMD_col = IMD_col).reset_index()
    return pivot

def get_AF_CI(df, observable, 
              q = 0.025, n = 100,
              eth_col = "Ethnicity Group",
              IMD_col = "IMD Quintile"):
    AF_bootstraps = np.asarray([new_bootstrap_AF(df, observable,
                    eth_col = eth_col, IMD_col = IMD_col).set_index(IMD_col).values for i in range(n)])
    
    CI_vals = np.round(np.quantile(AF_bootstraps, q, axis = 0),2)
    
    CI = pd.DataFrame(CI_vals, 
                      columns=new_bootstrap_AF(df, observable,
                   eth_col = eth_col, IMD_col = IMD_col).set_index(IMD_col).columns)
    CI[IMD_col] = ["1", "2", "3+"]
    cols = list(CI.columns)
    cols = [cols[-1]] + cols[0:-1]
    CI = CI[cols]
    return CI


def calc_AF(df, observable, n = 1000, 
            ref_IMD = "3+", ref_eth = "White",
            eth_col = "Ethnicity Group",
            IMD_col = "IMD Quintile"):
    
    AF = get_AF(df, 
                observable,
                eth_col = eth_col, 
                IMD_col = IMD_col).reset_index()
    
    lower = get_AF_CI(df, observable,  q = 0.025, n=n,
                      eth_col = eth_col, IMD_col = IMD_col)[list(AF.columns)]
    
    upper = get_AF_CI(df, observable, q = 0.975, n=n,
                      eth_col = eth_col, IMD_col = IMD_col)[list(AF.columns)]
    
    return [AF, lower, upper]

def calc_errors(AF):
    x = AF[0].columns.values[1:]
    y = AF[0].values[:,0]
    mids = AF[0].values[:,1:].astype(float)
    upper_err = (AF[0].values[:,1:] - AF[1].values[:,1:]).astype(float)
    lower_err = (AF[2].values[:,1:] - AF[0].values[:,1:]).astype(float)
    AF_errs = {"values" : mids, 
               "lower_err" : lower_err, 
               "upper_err" : upper_err, 
               "eths" : x, 
               "IMD vals" : y}
    return AF_errs

def plot_AF(AF_errs, outcome_counts, error_range = 100, 
            insuff_text_size = 12, tick_size = 14):
    
    # Hide values that didn't have enough data
    for i, eth in enumerate(AF_errs["eths"]):
        for j, imd in enumerate(AF_errs["IMD vals"]):
            insuf_data = np.isinf(AF_errs["values"][j, i]) or \
                any(np.isnan([AF_errs["lower_err"][j, i], AF_errs["upper_err"][j, i] ])) or \
                    abs(AF_errs["upper_err"][j, i] + AF_errs["lower_err"][j, i]) > error_range
            if insuf_data:
                AF_errs["values"][j,i] = np.nan
                
                
    fig = plt.figure(figsize = (10,4))
    ax = fig.add_subplot(111)
    ax.imshow(AF_errs["values"], cmap = "RdBu_r", 
              vmax = 130, vmin=-130, aspect = 0.8)
    AF_errs["eths"][AF_errs["eths"] == 'Middle Eastern'] = 'Middle\nEastern'
    ax.set_xticks(range(len(AF_errs["eths"])))
    ax.set_xticklabels(AF_errs["eths"], size=tick_size)
    ax.set_yticks(range(len(AF_errs["IMD vals"])))
    ax.set_yticklabels(AF_errs["IMD vals"], size=tick_size)
    
    # Create grid
    ax.set_xticks(ax.get_xticks() + 0.5, minor=True)
    ax.set_yticks(ax.get_yticks() + 0.5, minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    
    
    # annotate using nested crimes 
    outcome_counts = outcome_counts.values
    for i, eth in enumerate(AF_errs["eths"]):
        for j, imd in enumerate(AF_errs["IMD vals"]):
            if not ((i==len(AF_errs["eths"]) - 1) and (j==len(AF_errs["IMD vals"]) - 1)):
                # if value not infinite plot result
                if not np.isnan(AF_errs["values"][j, i]):
                    new_label = "$" + str(int(round(AF_errs["values"][j, i],0))) + "$%"
                    # Add AF %
                    ax.annotate(new_label, (i, j-0.15),
                                ha='center', size = 18)
                    
                    new_error = "($" + str(int(round(AF_errs["values"][j, i] - AF_errs["lower_err"][j, i],0))) \
                        + "$% to $" + str(int(round(AF_errs["values"][j, i] + AF_errs["upper_err"][j, i],0))) + "$%)"
                    # Add AF %
                    ax.annotate(new_error, (i, j+0.05),
                                ha='center', size = 12)

                    
                    n_ij = int(np.round(outcome_counts[j, i] * AF_errs["values"][j, i]/100, 0))
                    ax.annotate("[A={}]".format(n_ij), (i,j+.3), ha='center', 
                                size = 12)
                else:
                    ax.annotate("Insufficient\ndata", (i, j+0.16), 
                                size = insuff_text_size, ha='center')
            else:
                ax.annotate("REF", (i,j+.1), size = 12, ha='center')
    return fig


def calc_AF_rubin(df, observable, imp_col, n = 100, 
            ref_IMD = "3+", ref_eth = "White",
            eth_col = "Ethnicity Group",
            IMD_col = "IMD Quintile"):
    # https://www.missingdata.nl/missing-data/missing-data-methods/multiple-imputation/
    
    # Calculate AF for the whole imputed dataset
    AF_main = get_AF(df, 
                observable,
                eth_col = eth_col, 
                IMD_col = IMD_col).reset_index()
    
    # Seperate the dataset into the different imputations
    imputation_indicies = np.unique(df[imp_col])
    M = len(imputation_indicies)
    df_list = [df[df[imp_col] == imp_i] for imp_i in imputation_indicies]

    # Var-within
    var_list = []
    print(" Imputation ({}): ".format(M), end = "")
    for i, df_i in enumerate(df_list):
        # Calculate standard deviation of n bootstrap distributions
        bootstrap_i = [new_bootstrap_AF(df_i, observable, 
                         ref_IMD = ref_IMD, 
                         ref_eth = ref_eth,
                         eth_col = eth_col,
                         IMD_col = IMD_col) for j in range(n)]

        # Label each imputation
        for j in range(len(bootstrap_i)) :
            bootstrap_i[j]["boot_col"] = i
        
        # Combine bootstrapped data and calculate variance
        AF_comb = pd.concat(bootstrap_i, ignore_index=True, axis=0)
        var_i = AF_comb.groupby([IMD_col])[AF_comb.columns[1:-1]].var().reset_index()

        var_list.append(var_i)
        
        # Print progress
        print(i+1, end = ", ")
    print()
    
    ## Calcualte var sum ##
    
    # Label each imputation
    for j in range(len(var_list)) :
        var_list[j][imp_col] = j
        
    var_comb = pd.concat(var_list, ignore_index=True, axis=0)
    #print("\n",var_comb.columns)
    var_within = var_comb.groupby([IMD_col])[var_comb.columns[1:-1]].mean().reset_index()
    #print(var_within)

    # Var between
    # Set IMD col as index to hide it from subtraction calculation
    beta_diffs = [AF_main.set_index(IMD_col).sub(get_AF(df_i, 
                observable))**2 for df_i in df_list]


    #### START FROM HERE #### 
    # Label each imputation and add IMD back as regular column
    for j in range(len(beta_diffs)) :
        beta_diffs[j] = beta_diffs[j].reset_index()
        beta_diffs[j][imp_col] = j
    
    # Combine the list of square beta diffs
    beta_comb = pd.concat(beta_diffs, ignore_index=True, axis=0)
    

    beta_diff_sum = beta_comb.groupby([IMD_col])[beta_comb.columns[1:-1]].sum()

    var_between = beta_diff_sum / (M - 1)
    
    # calculate total variance   
    var_total = var_within.set_index(IMD_col) + var_between + var_between / M
 
    # Calculate confidence intervals
    lower = AF_main.set_index(IMD_col) - 1.96 * np.sqrt(var_total) / 2
    upper = AF_main.set_index(IMD_col) + 1.96 * np.sqrt(var_total) / 2
    
    return AF_main, lower.reset_index(), upper.reset_index()