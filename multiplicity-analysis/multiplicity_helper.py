#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:41:01 2023

@author: carollong
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
sys.path.append('../baseline-methods')
from hsls_utils import *

#------------------------------------------------------------
# processing model output pkl files
def process_pkl_no_eps(score_method):
    # score_method: [ [],[], []], each list from a random seed. num_itr x [] score_list
    # score_method[seed][eps_key] -> [],[],[] #itr x score_list
    # reorganize into: [[from itr1], [from itr2], []], each list from a itr. []: num_seed x score
    num_itr = len(score_method[0])
    score_compiled = [[] for i in range(num_itr)]
    # reorganize 
    for score_seed in score_method:
        for i in range(num_itr):
            score_compiled[i].append(score_seed[i])  
    return score_compiled

def process_pkl_w_eps(score_dic_list):
    # score_dic [{eps1: [], eps2: []}, {}] each epsilon has a dic of score_method as formatted above
    # regornanize as {eps1:[[from itr1], [from itr2], [from itr3]], eps2: }, each list from a itr. []: num_seed x score_list
    eps_list = score_dic_list[0].keys()
    eps_score_dic = {}

    for eps in eps_list:
        # initialize score_method to use process_pkl_no_eps() as subroutine
        score_method = []
        for seed_dic in score_dic_list:
            score_method.append(seed_dic[eps])
        score_compiled = process_pkl_no_eps(score_method)
        eps_score_dic[eps] = score_compiled
    return eps_score_dic

def pool_models_per_itr(score_dic_list):
    # score_dic [{eps1: [], eps2: []}, {}] each epsilon has a dic of score_method as formatted above
    # regornanize as [[from itr1], [from itr2], []], each list from a itr. []: num_model x score. num_models = num_seed x num_eps
    eps_list = list(score_dic_list[0].keys())
    num_itr = len(score_dic_list[0][eps_list[0]])
    scores_per_itr = [[] for i in range(num_itr)]
    
    for eps in eps_list:
        # initialize score_method to use process_pkl_no_eps() as subroutine
        score_method = []
        for seed_dic in score_dic_list:
            score_method.append(seed_dic[eps])
        score_compiled = process_pkl_no_eps(score_method)
        # add models to respective itr list
        for i in range(num_itr):
            for score_list in score_compiled[i]:
                scores_per_itr[i].append(score_list)
    return scores_per_itr


#------------------------------------------------------------
# get fairness metrics, from Fair Projection code

def confusion(y, y_pred):
    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

def get_statistical_parity_difference(y, s):
    sp0 = y[s==0].mean()
    sp1 = y[s==1].mean()
    return np.abs(sp1-sp0)

def get_meo_abs_diffs(y, y_pred, s):
    y0, y1 = y[s==0], y[s==1]
    y_pred0, y_pred1 = y_pred[s==0], y_pred[s==1]

    tpr0, fpr0 = confusion(y0, y_pred0)
    tpr1, fpr1 = confusion(y1, y_pred1)

    tpr_diff = tpr1 - tpr0
    fpr_diff = fpr1 - fpr0

    return (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2

#------------------------------------------------------------
# prep data for plots

# Note: only supporting 1 protected attribute.
# dataframe with columns: protected attribute, Y_test
def get_pred_table(data, itr_number):
    path = '../data/'
    if data == "enem":
        df = pd.read_pickle(path+'ENEM/enem-50000-20.pkl')
        label_name = 'gradebin'
        protected_attr = ['racebin']
        label_name = 'gradebin'
        df[label_name] = df[label_name].astype(int)    
    elif data =="hsls":
        file = 'HSLS/hsls_knn_impute.pkl'
        df = load_hsls_imputed(path, file, [])
        privileged_groups = [{'racebin': 1}]
        unprivileged_groups = [{'racebin': 0}]
        protected_attr = ['racebin']
        label_name = 'gradebin'
        
    dataset_orig_train, dataset_orig_vt = train_test_split(df, test_size=0.3, random_state=itr_number)    
    dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5, random_state=itr_number)    
    
    X_test = dataset_orig_test
    Y_test = dataset_orig_test[label_name]
    # test dataset with group attributes and multiplicity 
    pred_table = pd.DataFrame(data=X_test, copy=True)
    pred_table = pred_table[protected_attr]
    pred_table["y"] = Y_test
    return pred_table, protected_attr[0]

# return meo of every model in a iteration
def get_eo_acc(data, scores_per_itr, itr_number):
    meo_list = []
    acc_list = []
    pred_table, protected_attr = get_pred_table(data, itr_number)
    for scores in scores_per_itr:
        y_pred = (np.array(scores) > 0.5).astype('int')
        pred_table["y_pred"] = y_pred
        protected_arr = np.array(pred_table[protected_attr])
        y = pred_table['y']
        y_pred = pred_table['y_pred']
        meo = get_meo_abs_diffs(y, y_pred, protected_arr)
        meo_list.append(meo)
        acc = accuracy_score(y, y_pred)
        acc_list.append(acc)
    return meo_list, acc_list

def get_bin_label(meo_list,acc_list, x_edges, y_edges):
    x_list = meo_list
    y_list = acc_list
    # the number of bins in each dimension
    num_bins = 8
    
    # Define range of bins
    x_range = (0, 0.3)
    y_range = (0.615, 0.67)
    # Calculate the bin edges in each dimension
    x_bins = np.linspace(x_range[0], x_range[1], num_bins + 1)
    y_bins = np.linspace(y_range[0], y_range[1], num_bins + 1)
    
    # Bin the x and y values
    x_bin_indices = np.digitize(x_list, x_bins)
    y_bin_indices = np.digitize(y_list, y_bins)
    
    # Calculate the bin labels for each point
    bin_labels = (x_bin_indices - 1) * num_bins + y_bin_indices
    return bin_labels

def group_model_in_bin(scores_per_itr,bin_labels):
    # a dictionary where key is bin, value is a list of models in that bin
    bin_scores_dic = {label: [elem for elem, label_elem in zip(scores_per_itr, bin_labels) if label_elem == label] for label in set(bin_labels)}
    return bin_scores_dic

def get_score_std_for_bin(bin_number, bin_scores_dic):
    # compute score_std for a given bin
    score_list = bin_scores_dic[bin_number]
    xlen = np.array(bin_scores_dic[bin_number]).shape[0]
    ylen = np.array(bin_scores_dic[bin_number]).shape[1]
    score_list_reshaped = np.reshape(score_list, (xlen, ylen))
    score_sd_per_sample = pd.DataFrame(score_list_reshaped).std()
    return score_sd_per_sample

# =============================================================================
#def bin_models_per_itr(scores_per_itr):
    # put everything together
    
# =============================================================================


def compute_diff_multiplicity_per_itr(score_list, data, itr_number, quantile):
    path = '../data/'
    if data == "enem":
        df = pd.read_pickle(path+'ENEM/enem-50000-20.pkl')
        label_name = 'gradebin'
        #protected_attrs = ['racebin', 'sexbin']
        protected_attrs = ['racebin']
        label_name = 'gradebin'
        df[label_name] = df[label_name].astype(int)    
    
    dataset_orig_train, dataset_orig_vt = train_test_split(df, test_size=0.3, random_state=itr_number)    
    dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5, random_state=itr_number)    
    
    X_test = dataset_orig_test
    Y_test = dataset_orig_test[label_name]
    # test dataset with group attributes and multiplicity 
    pred_table = pd.DataFrame(data=X_test, copy=True)
    pred_table = pred_table[protected_attrs]
    
    # compute score_std
    xlen = np.array(score_list).shape[0]
    ylen = np.array(score_list).shape[1]
    score_list_1 = np.reshape(score_list, (xlen, ylen))
    score_sd_per_sample = pd.DataFrame(score_list_1).std()
    pred_table["group score std"] = np.array(score_sd_per_sample)
    
    score_std_gp_max = pred_table.groupby(protected_attrs).quantile(quantile).loc[:,"group score std"].max()
    score_std_gp_min = pred_table.groupby(protected_attrs).quantile(quantile).loc[:,"group score std"].min()
    abs_diff_score_std = abs(score_std_gp_max - score_std_gp_min)

    # compute ambiguity
    num_models = len(score_list)
    # threshold scores
    for i in range(num_models):
        # y_pred = (y_prob[:, 1] > t).astype('int')
        score_list[i] = [0 if score<0.5 else 1 for score in score_list[i]]
        score_list[i] = np.array(score_list[i])
        score_list[i] = score_list[i].astype(int)
        
    ambiguity = score_list[0]^score_list[1]
    for i in range(num_models):
        for j in range(i+1,num_models):
            if (i,j)==(0,1):
                continue
            ambiguity += score_list[i]^score_list[j]
    ambiguity = (ambiguity >= 1)
    
    pred_table["group ambiguity"] = ambiguity
    pred_table = pred_table.groupby(protected_attrs).mean()
    ambiguity_gp_max = pred_table.loc[:,"group ambiguity"].max()
    ambiguity_gp_min = pred_table.loc[:,"group ambiguity"].min()
    abs_diff_ambiguity = abs(ambiguity_gp_max - ambiguity_gp_min)
    
    return abs_diff_ambiguity, abs_diff_score_std

def compute_ambiguity_per_itr(score_list):
    num_models = len(score_list)
    # threshold scores
    for i in range(num_models):
        score_list[i] = [0 if score<0.5 else 1 for score in score_list[i]]
        score_list[i] = np.array(score_list[i])
        score_list[i] = score_list[i].astype(int)
        
    ambiguity = score_list[0]^score_list[1]
    for i in range(num_models):
        for j in range(i+1,num_models):
            if (i,j)==(0,1):
                continue
            ambiguity += score_list[i]^score_list[j]
    ambiguity_list = (ambiguity >= 1).astype(int)    
    ave_ambiguity_per_sample = np.average(ambiguity_list)  
    return ave_ambiguity_per_sample

def compute_ambiguity_list(score_list):
    # list [[from itr1], [from itr2], ...]
    ambiguity_list = []
    for itr in range(len(score_list)):
        ambiguity = compute_ambiguity_per_itr(score_list[itr])
        ambiguity_list.append(ambiguity)
    return ambiguity_list

def compute_diff_multiplicity_list(score_list, quantile):
    # list [[from itr1], [from itr2], ...]
    ambiguity_diff_list = []
    score_std_diff_list = []
    for itr in range(len(score_list)):
        ambiguity_diff, score_std_diff = compute_diff_multiplicity_per_itr(score_list[itr], "enem", itr, quantile)
        ambiguity_diff_list.append(ambiguity_diff)
        score_std_diff_list.append(score_std_diff)
    return ambiguity_diff_list, score_std_diff_list

def compute_ambiguity_dic(eps_score_dic):
    # dic {eps1:[[from itr1], [from itr2], [from itr3]], eps2:...} 
    # use compute_ambiguity_list as subroutine for each epsilon
    ambiguity_dic = {}
    for eps in eps_score_dic.keys():
        ambiguity_dic[eps] = compute_ambiguity_list(eps_score_dic[eps])
    return ambiguity_dic

def compute_diff_multiplicity_dic(eps_score_dic, quantile):
    # dic {eps1:[[from itr1], [from itr2], [from itr3]], eps2:...} 
    # use compute_ambiguity_list as subroutine for each epsilon
    ambiguity_dic = {}
    score_std_dic = {}
    for eps in eps_score_dic.keys():
        ambiguity_dic[eps], score_std_dic[eps] = compute_diff_multiplicity_list(eps_score_dic[eps], quantile)
    return ambiguity_dic,score_std_dic


def compute_score_sd_list(score_list, quantile):
    # compute average of score sd
    num_models = len(score_list)
    score_sd_list = []
    for itr in range(num_models):
        score_sd_per_sample = pd.DataFrame(score_list[itr]).std()
        top_percentile = score_sd_per_sample.quantile(quantile) 
        score_sd_list.append(top_percentile)
    return score_sd_list

def compute_score_sd_dic(eps_score_dic, quantile):
    ave_score_sd_dic = {}
    for eps in eps_score_dic.keys():
        ave_score_sd_dic[eps] = compute_score_sd_list(eps_score_dic[eps], quantile)
    return ave_score_sd_dic

