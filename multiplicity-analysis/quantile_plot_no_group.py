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


data = "enem"

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

# plot score_std quantile for fair and unfair bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(10):
    meo_list, acc_list = get_eo_acc(data, score_reduction[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    bin_scores_dic = {label: [elem for elem, label_elem in zip(score_reduction[i], bin_labels) if label_elem == label] for label in set(bin_labels)}
    
    # fair bins
    score_list_fair = []
    for bin_num in bins:
        if bin_num in bin_scores_dic.keys():
            #print(len(bin_scores_dic[bin_num]))
            for model in bin_scores_dic[bin_num]:
                score_list_fair.append(np.squeeze(model))
        #else:
            #print("no model in bin {}  for itr".format(str(bin_num)) + str(i))
    if len(score_list_fair)<5:
        print("number of models in bin {} for iteration {} is {} and <5".format(str(bins),i,len(score_list_fair)))
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

t = np.linspace(0,1,100)
# get the corresponding eo_range and acc_range for the bins
x_bin_start = x_bins[int(bins[0] / num_bins)]
x_bin_end =  x_bins[int(bins[0] / num_bins)]  + (int(bins[-1] / num_bins) - int(bins[0] / num_bins) + 1) * (x_bins[-1]-x_bins[-2])
y_bin_start = y_bins[bins[0] % num_bins] 
y_bin_end = y_bins[bins[0] % num_bins]  + (bins[-1] % num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
  
EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Reduction High Fairness \n(Acc:{:.2f}, EO violation:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)

# low fairness bins
if data == "enem":
    bins = [30,31, 38,39]
elif data =="hsls":
    bins = [19,20,21,22]
# plot score_std quantile for low-fairness bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(10):
    meo_list, acc_list = get_eo_acc(data, score_reduction[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    bin_scores_dic = {label: [elem for elem, label_elem in zip(score_reduction[i], bin_labels) if label_elem == label] for label in set(bin_labels)}
    
    # fair bins
    score_list_fair = []
    for bin_num in bins:
        if bin_num in bin_scores_dic.keys():
            #print(len(bin_scores_dic[bin_num]))
            for model in bin_scores_dic[bin_num]:
                score_list_fair.append(np.squeeze(model))
        #else:
            #print("no model in bin {}  for itr".format(str(bin_num)) + str(i))
    if len(score_list_fair)<5:
        print("number of models in bin {} for iteration {} is {} and <5".format(str(bins),i,len(score_list_fair)))
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

t = np.linspace(0,1,100)
# get the corresponding eo_range and acc_range for the bins
x_bin_start = x_bins[int(bins[0] / num_bins)]
x_bin_end =  x_bins[int(bins[0] / num_bins)]  + (int(bins[-1] / num_bins) - int(bins[0] / num_bins) + 1) * (x_bins[-1]-x_bins[-2])
y_bin_start = y_bins[bins[0] % num_bins] 
y_bin_end = y_bins[bins[0] % num_bins]  + (bins[-1] % num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
  
EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Reduction Low Fairness \n(Acc:{:.2f}, EO violation:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)



if data == "enem":
    bins = [62,63]
elif data =="hsls":
    bins = [29,36,37,38,39,44,45]
# plot score_std quantile for low-fairness bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(10):
    meo_list, acc_list = get_eo_acc(data, score_original[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    bin_scores_dic = {label: [elem for elem, label_elem in zip(score_original[i], bin_labels) if label_elem == label] for label in set(bin_labels)}
    
    # fair bins
    score_list_fair = []
    for bin_num in bins:
        if bin_num in bin_scores_dic.keys():
            #print(len(bin_scores_dic[bin_num]))
            for model in bin_scores_dic[bin_num]:
                score_list_fair.append(np.squeeze(model))
        #else:
            #print("no model in bin {}  for itr".format(str(bin_num)) + str(i))
    if len(score_list_fair)<5:
        print("number of models in bin {} for iteration {} is {} and <5".format(str(bins),i,len(score_list_fair)))
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

t = np.linspace(0,1,100)
# get the corresponding eo_range and acc_range for the bins
x_bin_start = x_bins[int(bins[0] / num_bins)]
x_bin_end =  x_bins[int(bins[0] / num_bins)]  + (int(bins[-1] / num_bins) - int(bins[0] / num_bins) + 1) * (x_bins[-1]-x_bins[-2])
y_bin_start = y_bins[bins[0] % num_bins] 
y_bin_end = y_bins[bins[0] % num_bins]  + (bins[-1] % num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
  
EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Baseline \n(Acc:{:.2f}, EO violation:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)


plt.xlabel('Standard Deviation of Scores')
plt.ylabel('Quantiles')
title = '{} Score Std Red'.format(data) 
#plt.title(title)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
#plt.show()
plt.savefig(title +'w_baseline_no_group.pdf', dpi=300)

print(title)
