#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:37:13 2023

@author: carollong
"""

from multiplicity_helper import *
from plot_figures_integrated import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = "enem"


score_original,  score_reduction = process_red_original_scores_per_itr(data, "rf", fair='eo', start_seed = 33, end_seed = 232)


 
num_itr = 10
num_ensem = 25
# for ensem 1.. 30, calculate 90% quantile of standard deviation using score_reduction
# Define the number of bins in each dimension
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
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
    bins = [5,6, 12, 13,14]
elif data =="hsls":
    bins = [10,11,12,13]

# get the scores of fair models (in the above fair bin) in all iterations 
score_list_fair_all_itr = [] #shape: num_itr * num_model* len(scores)
for i in range(num_itr):
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
    score_list_fair = np.squeeze(score_list_fair) # num_model* len(scores)
    score_list_fair_all_itr.append(score_list_fair)
    

# plot 90% score_std quantile for fair models, use 10 data splits to get error bar
num_models_in_ensem = [i+1 for i in range(num_ensem)]
mean_score_std_ensemble = []
std_score_std_ensemble = []
ensembled_eo = []
ensembled_acc = []
for j in num_models_in_ensem:
    # 90% score_std per iteration
    v_plot_list = []     
    for i in range(num_itr):
            fair_subset = np.array(score_list_fair_all_itr[i][0:10*j])
            temp = np.reshape(fair_subset, (10,j,-1))
            ensembled_scores = np.average(temp,axis = 1)
            if j == 25:
                temp_eo, temp_acc = get_eo_acc(data, ensembled_scores, i)
                ensembled_eo.append(temp_eo)
                ensembled_acc.append(temp_acc)
            # get 90/% score std
            score_sd_per_sample = pd.DataFrame(ensembled_scores).std()
            v_plot_list.append(score_sd_per_sample.quantile(0.9))
            
    mean_score_std_ensemble.append(np.mean(v_plot_list, axis = 0))
    std_score_std_ensemble.append(np.std(v_plot_list, axis=0, ddof = 1))

# get the corresponding eo_range and acc_range for the bins
x_bin_start = x_bins[int(bins[0] / num_bins)]
x_bin_end =  x_bins[int(bins[0] / num_bins)]  + (int(bins[-1] / num_bins) - int(bins[0] / num_bins) + 1) * (x_bins[-1]-x_bins[-2])
y_bin_start = y_bins[bins[0] % num_bins] 
y_bin_end = y_bins[bins[0] % num_bins]  + (bins[-1] % num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])

EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2

num_models_in_ensem = np.array(num_models_in_ensem)
mean_score_std_ensemble = np.array(mean_score_std_ensemble)
std_score_std_ensemble = np.array(std_score_std_ensemble)

ax.plot( num_models_in_ensem, mean_score_std_ensemble, label = "Reduction (Acc:{:.2f}, EO violation:{:.2f})".format(Acc_level,EO_level))
ax.fill_between(num_models_in_ensem, mean_score_std_ensemble+ std_score_std_ensemble, mean_score_std_ensemble - std_score_std_ensemble, alpha= 0.5)


# baseline 
# plot baseline std score 90% value
# =============================================================================
# v_plot_list = []     
# for i in range(num_itr):
#         baseline_subset = np.array(score_original[i][0:10])
#         # get 90/% score std
#         score_sd_per_sample = pd.DataFrame(baseline_subset).std()
#         v_plot_list.append(score_sd_per_sample.quantile(0.9))
# 
# v_plot_mean = np.mean(v_plot_list,axis = 0) 
# v_plot_std = np.std(v_plot_list, axis = 0, ddof=1) 
# 
# plt.axhline(y = v_plot_mean, color = 'b', linestyle = ':', label = "Baseline")
# 
# =============================================================================

if data == "enem":
    bins = [62,63]
elif data =="hsls":
    bins = [29,36,37,38,39,44,45]
    
# plot score_std quantile for original-fairness bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(num_itr):
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
    score_list_reshaped = np.squeeze(score_list_fair)[0:10] # num_model* len(scores)
    score_sd_per_sample = pd.DataFrame(score_list_reshaped).std() # len(scores)*1
    #assert(score_sd_per_sample.shape == (7500,))
   
    # get percentile score_std for each group
    t = 0.9
    v = score_sd_per_sample.quantile(t)
    v_plot_list.append(v)
  
# plot mean and std across itr
v_plot_mean = np.mean(v_plot_list,axis = 0) # (100,2)
v_plot_std = np.std(v_plot_list, axis = 0, ddof=1) # (100, 2)

# get the corresponding eo_range and acc_range for the bins
x_bin_start = x_bins[int(bins[0] / num_bins)]
x_bin_end =  x_bins[int(bins[0] / num_bins)]  + (int(bins[-1] / num_bins) - int(bins[0] / num_bins) + 1) * (x_bins[-1]-x_bins[-2])
y_bin_start = y_bins[bins[0] % num_bins] 
y_bin_end = y_bins[bins[0] % num_bins]  + (bins[-1] % num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
  
EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.axhline(y = v_plot_mean, color = 'orange', linestyle = ':', label = "Baseline (Acc:{:.2f}, EO violation:{:.2f})".format(Acc_level,EO_level))
ax.fill_between(num_models_in_ensem, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)


plt.xlabel('Number of Models in Ensemble')
plt.ylabel('90th Quantile Std Scores')

title = '{} Score Std Red of Ensemble '.format(data) 
#plt.title(title)
plt.legend(loc='upper right')
plt.tight_layout()
#plt.show()
plt.savefig(title +'.pdf', dpi=300)
