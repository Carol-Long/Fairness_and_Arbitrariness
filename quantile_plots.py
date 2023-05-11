#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:26:14 2023

@author: carollong
"""
from multiplicity_helper import *
from plot_figures_integrated import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = "enem"
path = './data/'
if data == "enem":
    df = pd.read_pickle(path+'ENEM/enem-50000-20.pkl')
    label_name = 'gradebin'
    #protected_attrs = ['racebin', 'sexbin']
    protected_attrs = ['racebin']
    label_name = 'gradebin'
    df[label_name] = df[label_name].astype(int)    

# fair_bin: 5+6; unfair_bin:39+40
#bins = [5]

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
#mega_bins = [[5,6],[13,14],[21,22],[30,31], [38,39]]
mega_bins = [[5,6],[30,31]]

for bins in mega_bins:
    original, hardt, reduction, rejection, leverage, mp, tolerance = load_enem_plots("rf", "eo", "33")
    score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp = process_scores_per_itr("enem", "rf", fair='eo', start_seed = 33, end_seed = 42)
    
    # Define the number of bins in each dimension
    num_bins = 8
    
    # Define the range of values in each dimension
    x_range = (0, 0.3)
    y_range = (0.615, 0.68)
    # Calculate the bin edges in each dimension
    x_bins = np.linspace(x_range[0], x_range[1], num_bins + 1)
    y_bins = np.linspace(y_range[0], y_range[1], num_bins + 1)
    
    # plot score_std quantile for fair and unfair bins, use 10 data splits to get error bar
    #percentile list per iteration
    v_plot_list = [] 
    for i in range(10):
        meo_list, acc_list = get_eo_acc("enem", score_reduction[i],i)
    
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
        assert(score_sd_per_sample.shape == (7500,))
        # get group attribute
        dataset_orig_train, dataset_orig_vt = train_test_split(df, test_size=0.3, random_state=i)    
        dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5, random_state=i)    
        
        X_test = dataset_orig_test
        # test dataset with group attributes and score_std per sample 
        pred_table = pd.DataFrame(data=X_test, copy=True)
        pred_table = pred_table[protected_attrs]
        pred_table["group score std"] = np.array(score_sd_per_sample)
        
        # get percentile score_std for each group
        t = np.linspace(0,1,100)
        v = [pred_table.groupby(protected_attrs).quantile(t_ix)['group score std'].to_numpy() for t_ix in t] # |t|* |protected_attrs|
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
  
#     for plotting various settings
    ax.plot(v_plot_mean[:,0],t, label = 'EO:[{:.3f},{:.3f}], Acc:[{:.3f},{:.3f}]'.format(x_bin_start,x_bin_end,y_bin_start,y_bin_end))
    ax.fill_betweenx( t,  v_plot_mean[:,0]+ v_plot_std[:,0],
                    v_plot_mean[:,0] - v_plot_std[:,0],  alpha=0.2)
# =============================================================================
#     ax.plot( v_plot_mean[:,1], t, label = "White")
#     ax.fill_betweenx(t, v_plot_mean[:,1]+ v_plot_std[:,1],
#                     v_plot_mean[:,1] - v_plot_std[:,1],  alpha=0.2)
# =============================================================================
    

plt.xlabel('score std')
plt.ylabel('cumulative dist')
title = 'Score Std of Random Forest Models with Reduction (Group: Non-White)'
plt.title(title)
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(title +'.pdf', dpi=300)


# plot for different bins same group
# =============================================================================
# ax.plot(v_plot_mean[:,0],t, label = 'EO:[{:.3f},{:.3f}], Acc:[{:.3f},{:.3f}]'.format(x_bin_start,x_bin_end,y_bin_start,y_bin_end))
# ax.fill_betweenx( t,  v_plot_mean[:,0]+ v_plot_std[:,0],
#                 v_plot_mean[:,0] - v_plot_std[:,0],  alpha=0.2)
# =============================================================================
# =============================================================================
#     ax.plot( v_plot_mean[:,1], t, label = 'EO:[{:.3f},{:.3f}], Acc:[{:.3f},{:.3f}]'.format(x_bin_start,x_bin_end,y_bin_start,y_bin_end))
#     ax.fill_betweenx(t, v_plot_mean[:,1]+ v_plot_std[:,1],
#                     v_plot_mean[:,1] - v_plot_std[:,1],  alpha=0.2)
# =============================================================================
    
# =============================================================================
# plt.xlabel('score std')
# plt.ylabel('cumulative dist')
# title = 'Score Std of Random Forest Models with Reduction (Group: Non-White)'
# plt.title(title)
# plt.legend()
# plt.tight_layout()
# #plt.show()
# plt.savefig(title +'.pdf', dpi=300)
# =============================================================================



# =============================================================================
# plt.plot(t, v_plot_mean[:,0], label = "NonWhite, Female")
# plt.fill_between( t,  v_plot_mean[:,0]+ v_plot_std[:,0],
#                 v_plot_mean[:,0] - v_plot_std[:,0],  alpha=0.2)
# 
# plt.plot(t, v_plot_mean[:,1], label = "NonWhite, Male")
# plt.fill_between(t, v_plot_mean[:,1]+ v_plot_std[:,1],
#                 v_plot_mean[:,1] - v_plot_std[:,1],  alpha=0.2)
# plt.plot(t, v_plot_mean[:,2], label = "White, Female")
# plt.fill_between( t,  v_plot_mean[:,2]+ v_plot_std[:,2],
#                 v_plot_mean[:,2] - v_plot_std[:,2],  alpha=0.2)
# 
# plt.plot(t, v_plot_mean[:,3], label = "White, Male")
# plt.fill_between( t,  v_plot_mean[:,3]+ v_plot_std[:,3],
#                 v_plot_mean[:,3] - v_plot_std[:,3],  alpha=0.2)
# 
# plt.ylabel('score std')
# plt.xlabel('cumulative dist')
# plt.legend()
# plt.title('Score Std Quantile of Fair Models')
# plt.legend()
# plt.show()
# =============================================================================



