#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:33:36 2023

@author: carollong
"""
from multiplicity_helper import *
from plot_figures_integrated import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# option: "enem", "hsls"
data = "enem"
# plot score_std quantile for fair and unfair bins, use 10 data splits to get error bar
# percentile list per iteration
def plot_quantile_score_std(data, method_score,bins):
    # method_score: score_reduction, score_rejection, etc
    v_plot_list = [] 
    bin_meo = []
    bin_acc = []
    for i in range(10):

        meo_list, acc_list = get_eo_acc(data, method_score[i],i)
    
        # Define the x and y data
        x_list = meo_list
        y_list = acc_list
    
        # Bin the x and y values
        x_bin_indices = np.digitize(x_list, x_bins)
        y_bin_indices = np.digitize(y_list, y_bins)
    
        # Calculate the bin labels for each point
        bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
        # option: score_reduction, score_rejection
        bin_scores_dic = {label: [(elem,meo,acc) for elem, meo, acc, label_elem 
                                  in zip(method_score[i], meo_list, acc_list, bin_labels) 
                                  if label_elem == label] for label in set(bin_labels)}
        # fair bins
        score_list_fair = []
        for bin_num in bins:
            if bin_num in bin_scores_dic.keys():
                #print(len(bin_scores_dic[bin_num]))
                for modeltuple in bin_scores_dic[bin_num]:
                    score_list_fair.append(np.squeeze(modeltuple[0]))
                    bin_meo.append(modeltuple[1])
                    bin_acc.append(modeltuple[2])
        # not enough model to calculate std
        if len(score_list_fair)<5:
            #print("reduction\n")
            #print("number of models in bin {} for iteration {} is {} and <5".format(str(bins),i,len(score_list_fair)))
            continue
        score_list_reshaped = np.squeeze(score_list_fair) # num_model* len(scores)
        score_sd_per_sample = pd.DataFrame(score_list_reshaped).std() # len(scores)*1
        #assert(score_sd_per_sample.shape == (7500,))
       
        # get percentile score_std for each group
        t = np.linspace(0,1,100)
        v = [score_sd_per_sample.quantile(t_ix) for t_ix in t] # |t|* |protected_attrs|
        v_plot_list.append(v)
      
    # print(np.array(v_plot_list).shape) -> (10, 100, 2)
    # plot mean and std across itr
    v_plot_mean = np.mean(v_plot_list,axis = 0) # (100,2)
    v_plot_std = np.std(v_plot_list, axis = 0, ddof=1) # (100, 2)
    EO_level = np.mean(bin_meo)
    Acc_level = np.mean(bin_acc)
    return t, v_plot_mean,v_plot_std,EO_level,Acc_level



fig, ax = plt.subplots(1, 1, figsize=(4, 3))
#mega_bins = [[5,6],[13,14],[21,22],[30,31], [38,39]]

if data=="enem":
    original, hardt, reduction, rejection, leverage, mp, tolerance = load_enem_plots("rf", "eo", "33")
elif data == "hsls":
    original, hardt, reduction, rejection, leverage, mp, tolerance = load_hsls_plots("rf", "eo", "33")
    
score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp = process_scores_per_itr(data, "rf", fair='eo', start_seed = 33, end_seed = 42)

# Define the number of bins in each dimension
num_bins = 8


# Define the range of values in each dimension
if data == "enem":
    x_range = (0, 0.3)
    y_range = (0.615, 0.68)
elif data =="hsls":
    x_range = (0, 0.3)
    y_range = (0.70, 0.765)
# Calculate the bin edges in each dimension
x_bins = np.linspace(x_range[0], x_range[1], num_bins + 1)
y_bins = np.linspace(y_range[0], y_range[1], num_bins + 1)


# high fairness bins
if data == "enem":
    bins = [5,6,13,14]
elif data =="hsls":
    bins = [10,11,12,13]

t, v_plot_mean,v_plot_std,EO_level,Acc_level = plot_quantile_score_std(data, score_reduction,bins)
ax.plot( v_plot_mean, t, label = "Reduction High Fair \n(Acc:{:.2f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)

# low fairness bins
if data == "enem":
    bins = [30,31, 38,39]
elif data =="hsls":
    bins = [19,20,21,22]
t, v_plot_mean,v_plot_std,EO_level,Acc_level = plot_quantile_score_std(data, score_reduction,bins)
ax.plot( v_plot_mean, t, label = "Reduction Low Fair \n(Acc:{:.2f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)



if data == "enem":
    bins = [62,63]
elif data =="hsls":
    bins = [29,36,37,38,39,44,45]
t, v_plot_mean,v_plot_std,EO_level,Acc_level = plot_quantile_score_std(data, score_original,bins)
ax.plot( v_plot_mean, t, label = "Baseline \n(Acc:{:.2f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)


plt.xlabel('Classifier Score Std.')
plt.ylabel('Quantiles of Score Std.')
title = '{} Score Std Red'.format(data) 
#plt.title(title)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
#plt.show()
plt.savefig(title +'w_baseline.pdf', dpi=300)
