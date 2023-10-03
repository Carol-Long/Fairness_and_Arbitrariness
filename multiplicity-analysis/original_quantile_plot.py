#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:52:33 2023

@author: carollong

Baseline vs Reduction (high and low fairness bin)
"""

from multiplicity_helper import *
from plot_figures_integrated import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# option: hsls/enem/adult
data = "enem"
fig, ax = plt.subplots(1, 1, figsize=(6.3, 3.5))
# option: logit, rf, gbm
model_base = 'rf'
score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp = process_scores_per_itr(data, model_base, fair='eo', start_seed = 33, end_seed = 42)
# threshold baseline
threshold = 0.5
score_threshold_original = [[np.where(scores >= threshold, 1, 0) for scores in sublist] for sublist in score_original]

# Define the number of bins in each dimension
num_bins = 8


# Define the range of values in each dimension
if data == "enem":
    if model_base =="rf":
        x_range = (0, 0.3)
        y_range = (0.615, 0.68)
    elif model_base == "gbm" or model_base == "logit":
        x_range = (0, 0.32)
        y_range = (0.615, 0.681)        
elif data =="hsls":
    x_range = (0, 0.3)
    y_range = (0.70, 0.765)
elif data =="adult":
    if model_base=="rf":
        x_range = (0, 0.07)
        y_range = (0.83, 0.865)
    elif model_base == "gbm":
        x_range = (0, 0.08)
        y_range = (0.84, 0.875)
    
# Calculate the bin edges in each dimension
x_bins = np.linspace(x_range[0], x_range[1], num_bins + 1)
y_bins = np.linspace(y_range[0], y_range[1], num_bins + 1)


# high fairness bins
if data == "enem":
    if model_base == "rf":
        bins = [5,6,13,14]
    elif model_base == "gbm":
        bins = [6,7,14,15]
    elif model_base == "logit":
        bins = [6,7,14,15]
elif data =="hsls":
    if model_base == "rf":
        bins = [10,11,12,13,14]
    elif model_base == "gbm":
        bins = [5,6,7]
    elif model_base == "logit":
        bins = [1,2,3,4]
elif data == "adult":
    if model_base == "rf":
        bins = [20,21,22,23,28,29,30,31]
    elif model_base == "gbm":
        bins = [12,13,20,21,28,28]


# plot score_std quantile for fair and unfair bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
bin_meo = []
bin_acc = []
for i in range(10):
    # option: score_reduction, score_rejection
    meo_list, acc_list = get_eo_acc(data, score_reduction[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    # option: score_reduction, score_rejection
    bin_scores_dic = {label: [(elem,meo,acc) for elem, meo, acc, label_elem in zip(score_reduction[i], meo_list, acc_list, bin_labels) if label_elem == label] for label in set(bin_labels)}
    
    # fair bins
    score_list_fair = []
    for bin_num in bins:
        if bin_num in bin_scores_dic.keys():
            #print(len(bin_scores_dic[bin_num]))
            for modeltuple in bin_scores_dic[bin_num]:
                score_list_fair.append(np.squeeze(modeltuple[0]))
                bin_meo.append(modeltuple[1])
                bin_acc.append(modeltuple[2])
        #else:
            #print("no model in bin {}  for itr".format(str(bin_num)) + str(i))
    if len(score_list_fair)<5:
        print("reduction\n")
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

# =============================================================================
# t = np.linspace(0,1,100)
# # get the corresponding eo_range and acc_range for the bins
# x_bin_start = x_bins[int(bins[0] / num_bins)]
# x_bin_end =  x_bins[int(bins[0] / num_bins)]  + (int(bins[-1] / num_bins) - int(bins[0] / num_bins) + 1) * (x_bins[-1]-x_bins[-2])
# y_bin_start = y_bins[bins[0] % num_bins-1] 
# if bins[-1] % num_bins == 0:
#     y_bin_end = y_bins[bins[0] % num_bins-1]  + (num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
# else:
#     y_bin_end = y_bins[bins[0] % num_bins-1]  + (bins[-1] % num_bins - bins[0] % num_bins+1) * (y_bins[-1]-y_bins[-2])
# 
# EO_level = (x_bin_start+x_bin_end)/2
# Acc_level = (y_bin_start+y_bin_end)/2
# =============================================================================
EO_level = np.mean(bin_meo)
Acc_level = np.mean(bin_acc)
ax.plot( v_plot_mean, t, label = "Reduction High Fair(Acc:{:.3f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, np.where(v_plot_mean - v_plot_std<0, 0, v_plot_mean - v_plot_std),  alpha=0.2)


# low fairness bins
if data == "enem":
    if model_base == "rf":
        bins = [30,31, 38,39]
    elif model_base == "gbm":
        bins = [23,24,31,32]
    elif model_base == "logit":
        bins = [23,24,31,32]
elif data =="hsls":
    if model_base == "rf":
        bins = [19,20,21,22]
    elif model_base == "gbm":
        bins = [13,14,15]
    elif model_base == "logit":
        bins = [10,11,12,19,20]
elif data == "adult":
    if model_base == "rf":
        bins = [37,38,39,45,46,47]
    elif model_base == "gbm":
        bins = [28,29,36,37,38,39,44,45,46]
    
# plot score_std quantile for low-fairness bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(10):
    # option: score_reduction, score_rejection
    meo_list, acc_list = get_eo_acc(data, score_reduction[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    # option: score_reduction, score_rejection
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
y_bin_start = y_bins[bins[0] % num_bins-1] 
if bins[-1] % num_bins == 0:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
else:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (bins[-1] % num_bins - bins[0] % num_bins+1) * (y_bins[-1]-y_bins[-2])

EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Reduction Low Fair(Acc:{:.3f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, np.where(v_plot_mean - v_plot_std<0, 0, v_plot_mean - v_plot_std),  alpha=0.2)


# rejection high fair
if data == "enem":
    if model_base == "rf":
        bins = [5,6]  

    
# plot score_std quantile for original-fairness bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(10):
    meo_list, acc_list = get_eo_acc(data, score_rejection[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    bin_scores_dic = {label: [elem for elem, label_elem in zip(score_rejection[i], bin_labels) if label_elem == label] for label in set(bin_labels)}
    
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
        print("rejection\n")
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
y_bin_start = y_bins[bins[0] % num_bins-1] 
if bins[-1] % num_bins == 0:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
else:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (bins[-1] % num_bins - bins[0] % num_bins+1) * (y_bins[-1]-y_bins[-2])

EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Rejection High Fair(Acc:{:.3f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, np.where(v_plot_mean - v_plot_std<0, 0, v_plot_mean - v_plot_std),  alpha=0.2)

# rejection low fair
if data == "enem":
    if model_base == "rf":
        bins = [54,55,62,63]  

    
# plot score_std quantile for original-fairness bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(10):
    meo_list, acc_list = get_eo_acc(data, score_rejection[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    bin_scores_dic = {label: [elem for elem, label_elem in zip(score_rejection[i], bin_labels) if label_elem == label] for label in set(bin_labels)}
    
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
        print("rejection\n")
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
y_bin_start = y_bins[bins[0] % num_bins-1] 
if bins[-1] % num_bins == 0:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
else:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (bins[-1] % num_bins - bins[0] % num_bins+1) * (y_bins[-1]-y_bins[-2])

EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Rejection Low Fair(Acc:{:.3f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, np.where(v_plot_mean - v_plot_std<0, 0, v_plot_mean - v_plot_std),  alpha=0.2)

# baseline

if data == "enem":
    if model_base == "rf":
        bins = [62,63]
    elif model_base == "gbm":
        bins = [55,56,63,64]
    elif model_base == "logit":
        bins = [55,56,63,64]
elif data =="hsls":
    #bins = [29,36,37,38,39,44,45]
    if model_base == "rf":
        bins = [35,36,37,38,39]
    elif model_base == "gbm":
        bins = [38,39,46,47]
    elif model_base == "logit":
        bins = [44,45]
elif data =="adult":  
    if model_base == "rf":
        bins = [45,46,47,48,53,54,55,56,61,62,63,64]
    elif model_base == "gbm":
        bins = [52,53,54,60,61,62]

    
# plot score_std quantile for original-fairness bins, use 10 data splits to get error bar
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
        print("baseline\n")
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
y_bin_start = y_bins[bins[0] % num_bins-1] 
if bins[-1] % num_bins == 0:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
else:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (bins[-1] % num_bins - bins[0] % num_bins+1) * (y_bins[-1]-y_bins[-2])

EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Baseline (Acc:{:.3f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, v_plot_mean - v_plot_std,  alpha=0.2)


# thresholded baseline
if data == "enem":
    if model_base == "rf":
        #bins = [54,55,62,63]    
        bins = [62,63]  
if data =="hsls":
    #bins = [29,36,37,38,39,44,45]
    if model_base == "rf":
        bins = [35,36,37,38,39]

    
# plot score_std quantile for original-fairness bins, use 10 data splits to get error bar
#percentile list per iteration
v_plot_list = [] 
for i in range(10):
    meo_list, acc_list = get_eo_acc(data, score_threshold_original[i],i)

    # Define the x and y data
    x_list = meo_list
    y_list = acc_list

    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)

    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    bin_scores_dic = {label: [elem for elem, label_elem in zip(score_threshold_original[i], bin_labels) if label_elem == label] for label in set(bin_labels)}
    
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
        print("thresholded baseline\n")
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
y_bin_start = y_bins[bins[0] % num_bins-1] 
if bins[-1] % num_bins == 0:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (num_bins - bins[0] % num_bins + 1) * (y_bins[-1]-y_bins[-2])
else:
    y_bin_end = y_bins[bins[0] % num_bins-1]  + (bins[-1] % num_bins - bins[0] % num_bins+1) * (y_bins[-1]-y_bins[-2])

EO_level = (x_bin_start+x_bin_end)/2
Acc_level = (y_bin_start+y_bin_end)/2
ax.plot( v_plot_mean, t, label = "Thresholded Baseline (Acc:{:.3f}, Mean EO:{:.2f})".format(Acc_level,EO_level))
ax.fill_betweenx(t, v_plot_mean+ v_plot_std, np.where(v_plot_mean - v_plot_std<0, 0, v_plot_mean - v_plot_std),  alpha=0.2)

plt.xlabel('Classifier Score Std.')
plt.ylabel('Quantiles of Score Std.')
title = '{} Score Std Red '.format(data) 
#plt.title(title)
plt.legend(loc='lower right')
plt.tight_layout()
#plt.show()
plt.savefig(title + model_base + 'high_low.pdf', dpi=300)

