#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:39:36 2023

@author: carollong
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

sns.set()
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from multiplicity_helper import *

def compute_missing_mean_se(data):
    mean, std = [], []
    for i in range(data.shape[0]):
        temp = []
        for j in range(data.shape[1]):
                    temp.append(data[i, j])
        temp = np.array(temp)
        mean.append(np.mean(temp))
        std.append(np.std(temp))
    return np.array(mean), np.array(std)/np.sqrt(data.shape[0])

def compute_mean_se(data, x='meo_mp', y='acc_mp'):
    acc_mp_mean, acc_mp_se = compute_missing_mean_se(data[y])
    fr_mp_mean, fr_mp_se = compute_missing_mean_se(data[x])
    return [fr_mp_mean, acc_mp_mean, fr_mp_se, acc_mp_se]

def plot_point(ax, data, x, y, label, marker, color, alpha):
    ax.errorbar(data[x+'_mean'], data[y+'_mean'], xerr=data[x+'_std']/np.sqrt(10), 
                yerr=data[y+'_std']/np.sqrt(10), marker=marker, label = label, color=color, 
                alpha = alpha, markersize=3, linewidth = 0.5)

def plot_multi(ax, data, x, y, idx, label, marker, color, alpha):
    ax.errorbar(np.array(data[x+'_mean'])[idx], np.array(data[y+'_mean'])[idx], 
                xerr=np.array(data[x+'_std'])[idx]/np.sqrt(10), 
                yerr=np.array(data[y+'_std'])[idx]/np.sqrt(10), marker=marker, label = label, 
                color=color, alpha = alpha, markersize=3, linewidth = 0.5)

def plot_leverage(ax, data, x, y, label, marker, color, alpha):
    ax.errorbar(data[x].mean(), data[y].mean(), 
                xerr=data[x].std()/np.sqrt(10), yerr=data[y].std()/np.sqrt(10), 
                marker=marker, label = label, color=color, alpha = alpha, markersize=3, linewidth = 0.5)
    return

def plot_mp(ax, data, x, y, idx, label, marker, color, alpha):
    mean_se = compute_mean_se(data, x, y)
    ax.errorbar(np.abs(mean_se[0][idx]), mean_se[1][idx], xerr=mean_se[2][idx]/np.sqrt(10), yerr=mean_se[3][idx]/np.sqrt(10), 
                marker=marker, label = label, color=color, markersize=3, alpha = alpha, linewidth = 0.5)
    return

def plot_base(ax, x, y, label, alpha):
    ax.plot(x, y, marker='*', linestyle = '', markersize=4, color='black', label=label, zorder=10, alpha = alpha, linewidth = 0.5)
    return 

def plot_point_pre_calc(ax, data, x, y, label, marker, color, alpha):
    ax.errorbar(data[x+'_mean'], data[y+'_mean'], xerr=data[x+'_std'], 
                yerr=data[y+'_std'], marker=marker, label = label, color=color, 
                alpha = alpha, markersize=4, linewidth = 0.8)

def plot_multi_pre_calc(ax, data, x, y, idx, label, marker, color, alpha):
    ax.errorbar(np.array(data[x+'_mean'])[idx], np.array(data[y+'_mean'])[idx], 
                xerr=np.array(data[x+'_std'])[idx], 
                yerr=np.array(data[y+'_std'])[idx], marker=marker, label = label, 
                color=color, alpha = alpha, markersize=4, linewidth = 0.8)


def plot_leverage_pre_calc1(ax, data, x, y, label, marker, color, alpha):
    ax.errorbar(data["meo"], data["ambiguity_mean"], 
                xerr=data["meo_std"], yerr=data["ambiguity_std"], 
                marker=marker, label = label, color=color, alpha = alpha, markersize=4, linewidth = 0.8)
    return

def plot_leverage_pre_calc2(ax, data, x, y, label, marker, color, alpha):
    ax.errorbar(data["meo"], data["score_std_mean"], 
                xerr=data["meo_std"], yerr=data["score_std_std"], 
                marker=marker, label = label, color=color, alpha = alpha, markersize=4, linewidth = 0.8)
    return

def get_model_name(model):
    if (model == 'rf') or (model == 'rfc'):
        model_name = 'Random Forest'
    elif model == 'logit':
        model_name = 'Logistic Regression'
    elif model == 'gbm':
        model_name = 'GBM'
        
    return model_name

def basic_plot(data, idx, model, ax=None, fair='eo', mp_name='ce', alpha = 0.5):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    
    mp_all = data['mp_all']
    
    if fair == 'eo':
        ## Hardt
        plot_point(ax, data['hardt'], 'abseo', 'acc', 'EqOdds', 'o', 'steelblue', alpha)
        ## rejection
        if 'rejection' in data.keys():
            plot_multi(ax, data['rejection'], 'abseo', 'acc', idx['rejection'], 'Rejection', 'o', 'gold', alpha)
        ## reduction
        plot_multi(ax, data['reduction'], 'abseo', 'acc', idx['reduction'], 'Reduction', 'o', 'coral', alpha)
        ## calibration
        if 'calibration' in data.keys():
            plot_point(ax, data['calibration'], 'abseo', 'acc', 'CalEqOdds', 'o', 'darkgoldenrod', alpha)
        ## Leveraging
        plot_leverage(ax, data['leverage'], 'meo', 'acc',  'LevEqOpp', 'o', 'darkolivegreen', alpha)
        ## Model Projection
        if model == 'rf':
            model = 'rfc'
        
        if mp_name == 'ce': 
            mp = mp_all[model+'_ce_meo']
            plot_mp(ax, mp, 'meo_abs', 'acc', idx['mp'], 'FairProjection-CE', 'o', 'darkred', alpha)
        elif mp_name == 'kl':
            mp = mp_all[model+'_kl_meo']
            plot_mp(ax, mp, 'meo_abs', 'acc', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        elif mp_name == 'all':
            mp = mp_all[model+'_ce_meo']
            plot_mp(ax, mp, 'meo_abs', 'acc', idx['mp'], 'FairProjection-CE', 'o', 'darkred', alpha)
            mp = mp_all[model+'_kl_meo']
            plot_mp(ax, mp, 'meo_abs', 'acc', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        ## Base
        if 'original' in data.keys():
            plot_base(ax, x = data['original']['abseo_mean'], y = data['original']['acc_mean'], label= 'Base', alpha = alpha)
        else:
            plot_base(ax, x=mp['meo_abs'].mean(axis=1)[idx['mp'][-1]], y=mp['acc'].mean(axis=1)[idx['mp'][-1]], label='Base', alpha = alpha)
    
    elif fair == 'eo_ambiguity':
        ## Hardt
        plot_point_pre_calc(ax, data['hardt'], 'abseo', 'ambiguity', 'EqOdds', 'o', 'steelblue', alpha)
        ## rejection
        if 'rejection' in data.keys():
            plot_multi_pre_calc(ax, data['rejection'], 'abseo', 'ambiguity', idx['rejection'], 'Rejection', 'o', 'gold', alpha)
        ## reduction
        plot_multi_pre_calc(ax, data['reduction'], 'abseo', 'ambiguity', idx['reduction'], 'Reduction', 'o', 'coral', alpha)
        ## calibration
        if 'calibration' in data.keys():
            plot_point_pre_calc(ax, data['calibration'], 'abseo', 'ambiguity', 'CalEqOdds', 'o', 'darkgoldenrod', alpha)
        ## Leveraging
        plot_leverage_pre_calc1(ax, data['leverage'], 'meo', 'ambiguity',  'LevEqOpp', 'o', 'darkolivegreen', alpha)

        ## Base
        if 'original' in data.keys():
            plot_base(ax, x = data['original']['abseo_mean'], y = data['original']['ambiguity_mean'], label= 'Base', alpha = alpha)
        else:
            plot_base(ax, x=mp['meo_abs'].mean(axis=1)[idx['mp'][-1]], y=mp['ambiguity'].mean(axis=1)[idx['mp'][-1]], label='Base', alpha = alpha)
    
    elif fair == 'eo_score_std':
        ## Hardt
        plot_point_pre_calc(ax, data['hardt'], 'abseo', 'score_std', 'EqOdds', 'o', 'steelblue', alpha)
        ## rejection
        if 'rejection' in data.keys():
            plot_multi_pre_calc(ax, data['rejection'], 'abseo', 'score_std', idx['rejection'], 'Rejection', 'o', 'gold', alpha)
        ## reduction
        plot_multi_pre_calc(ax, data['reduction'], 'abseo', 'score_std', idx['reduction'], 'Reduction', 'o', 'coral', alpha)
        ## calibration
        if 'calibration' in data.keys():
            plot_point_pre_calc(ax, data['calibration'], 'abseo', 'score_std', 'CalEqOdds', 'o', 'darkgoldenrod', alpha)
        ## Leveraging
        plot_leverage_pre_calc2(ax, data['leverage'], 'meo', 'score_std',  'LevEqOpp', 'o', 'darkolivegreen', alpha)
    
        ## Base
        if 'original' in data.keys():
            plot_base(ax, x = data['original']['abseo_mean'], y = data['original']['score_std_mean'], label= 'Base', alpha = alpha)
        else:
            plot_base(ax, x=mp['meo_abs'].mean(axis=1)[idx['mp'][-1]], y=mp['score_std'].mean(axis=1)[idx['mp'][-1]], label='Base', alpha = alpha)
                
    elif fair == 'maxeo':
        ## Hardt
        plot_point(ax, data['hardt'], 'maxeo', 'acc', 'EqOdds', 'o', 'orange')
        ## Leveraging
        plot_leverage(ax, data['leverage'], 'maxeo', 'acc',  'LevEqOpp', 'o', 'darkolivegreen')
        ## Model Projection
        if model == 'rf':
            model = 'rfc'
        
        if mp_name == 'ce': 
            mp = mp_all[model+'_ce_meo']
            plot_mp(ax, mp, 'mo', 'acc', idx['mp'], 'FairProjection-CE', 'o', 'darkred')
        elif mp_name == 'kl':
            mp = mp_all[model+'_kl_meo']
            plot_mp(ax, mp, 'mo', 'acc', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        elif mp_name == 'all':
            mp = mp_all[model+'_ce_meo']
            plot_mp(ax, mp, 'mo', 'acc', idx['mp'], 'FairProjection-CE', 'o', 'darkred')
            mp = mp_all[model+'_kl_meo']
            plot_mp(ax, mp, 'mo', 'acc', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        ## Base
        plot_base(ax, x=mp['mo'].mean(axis=1)[idx['mp'][-1]], y=mp['acc'].mean(axis=1)[idx['mp'][-1]], label='Base')

    elif fair == 'sp': 
        ## rejection
        if 'rejection' in data.keys():
            plot_multi(ax,  data['rejection'], 'sp', 'acc', idx['rejection'], 'Rejection', 'o', 'gold')
        ## reduction
        plot_multi(ax, data['reduction'], 'sp', 'acc', idx['reduction'], 'Reduction', 'o', 'coral')
        ## Model Projection
        if model == 'rf':
            model = 'rfc'
        if mp_name == 'ce': 
            mp = mp_all[model+'_ce_sp']
            plot_mp(ax, mp, 'sp', 'acc', idx['mp'], 'FairProjection-CE', 'o', 'darkred')
        elif mp_name == 'kl':
            mp = mp_all[model+'_kl_sp']
            plot_mp(ax, mp, 'sp', 'acc', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        elif mp_name == 'all':
            mp = mp_all[model+'_ce_sp']
            plot_mp(ax, mp, 'sp', 'acc', idx['mp'], 'FairProjection-CE', 'o', 'darkred')
            mp = mp_all[model+'_kl_sp']
            plot_mp(ax, mp, 'sp', 'acc', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        ## Base
        
        if 'original' in data.keys():
            plot_base(ax, x = data['original']['sp_mean'], y = data['original']['acc_mean'], label= 'Base')
        else:
            plot_base(ax, x=mp['sp'].mean(axis=1)[idx['mp'][-1]], y=mp['acc'].mean(axis=1)[idx['mp'][-1]], label='Base')
                
    elif fair == 'brier': 
        ## rejection
        plot_multi(ax,  data['rejection'], 'sp', 'brier', idx['rejection'], 'Rejection', 'o', 'gold')
        ## reduction
        plot_multi(ax, data['reduction'], 'sp', 'brier', idx['reduction'], 'Reduction', 'o', 'orangered')
        ## Model Projection
        if model == 'rf':
            model = 'rfc'
        if mp_name == 'ce': 
            mp = mp_all[model+'_ce_sp']
            plot_mp(ax, mp, 'sp', 'brier', idx['mp'], 'FairProjection-CE', 'o', 'darkred')
        elif mp_name == 'kl':
            mp = mp_all[model+'_kl_sp']
            plot_mp(ax, mp, 'sp', 'brier', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        elif mp_name == 'all':
            mp = mp_all[model+'_ce_sp']
            plot_mp(ax, mp, 'sp', 'brier', idx['mp'], 'FairProjection-CE', 'o', 'darkred')
            mp = mp_all[model+'_kl_sp']
            plot_mp(ax, mp, 'sp', 'brier', idx['mp'], 'FairProjection-KL', 'x', 'darkred')
        ## Base
        plot_base(ax, x=mp['sp'].mean(axis=1)[idx['mp'][-1]], y=mp['brier'].mean(axis=1)[idx['mp'][-1]], label='Base')

    ## Figure configs
    
    if fair == 'eo':
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Mean Equalized Odds')
    elif fair == 'eo_ambiguity':
        ax.set_ylabel('Ambiguity')
        ax.set_xlabel('Mean Equalized Odds')
    elif fair == 'eo_score_std':
        ax.set_ylabel('Score Std')
        ax.set_xlabel('Mean Equalized Odds')        
    elif fair == 'maxeo':
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Max Equalized Odds')
    elif fair == 'sp':
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Statistical Parity')
        
def load_adult_plots(model, fair='eo'):
    ## benchmarks
    with open('../experimental-data/adult/results/eqodds_'+model+'_s42_.pkl', 'rb+') as f: 
        adult_eqodds = pickle.load(f)
    with open('../experimental-data/adult/results/original_'+model+'_s42_.pkl', 'rb+') as f: 
        adult_original = pickle.load(f)
    with open('../experimental-data/adult/results/reduction_'+model+'_s42_eo.pkl', 'rb+') as f: 
        adult_reduction_eo = pickle.load(f)
    with open('../experimental-data/adult/results/reduction_'+model+'_s42_sp.pkl', 'rb+') as f: 
        adult_reduction_sp = pickle.load(f) 
    with open('../experimental-data/adult/results/roc_'+model+'_s42_eo.pkl', 'rb+') as f: 
        adult_roc_eo = pickle.load(f)
    with open('../experimental-data/adult/results/roc_'+model+'_s42_sp.pkl', 'rb+') as f: 
        adult_roc_sp = pickle.load(f)
        
    with open('../experimental-data/adult/results/leveraging_'+model+'_s42_eo.pkl', 'rb+') as f: 
        adult_leverage = pickle.load(f)

    ## Model Projection
    with open('../experimental-data/adult/results/adult-mp-new.pkl', 'rb+') as f: 
        adult_mp = pickle.load(f)
        
    if model == 'rf':
        model = 'rfc'
   
    adult_tolerance = adult_mp['tolerance']
    
    if fair == 'eo':
        return adult_eqodds, adult_reduction_eo, adult_roc_eo, adult_leverage, adult_mp, adult_original, adult_tolerance
        
    else: 
        return None, adult_reduction_sp, adult_roc_sp, None, adult_mp, adult_original, adult_tolerance
        

def plot_adult(hardt, reduction, rejection, leverage, mp_all, original, tolerance, model, ax =None, fair='eo', mp_name='ce'):
    if fair == 'sp':
        idx_projection = np.array([4,5,6,7]) #np.arange(len(tolerance))
    else: 
        idx_projection = np.arange(len(tolerance))
    idx_rejection = np.arange(10)
    idx_reduction = np.array([0, 2, 3, 4, 5])
    idx_calibration = []
    idx_fact= [] 
    
    data = {'hardt': hardt, 'reduction': reduction, 'leverage': leverage,  'mp_all': mp_all, 
            'original': original, 'tolerance': tolerance}
    idx = {'calibration': idx_calibration, 'reduction': idx_reduction, 'rejection': idx_rejection, 'mp': idx_projection, 'fact': idx_fact }
    
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        
    basic_plot(data, idx, model, ax, fair, mp_name)
    
    ax.set_title(r"$\bf{Adult}$ (" + get_model_name(model) + ')')
    if ax == None:
        fig.savefig('plot/adult-'+model+'-' + mp_name,format='png', dpi=300)
   
def load_hsls_plots(model, fair='eo', seed = 42):
        
    with open('../experimental-data/hsls/results/eqodds_'+model+'_s{}_.pkl'.format(seed), 'rb+') as f: 
        hsls_eqodds = pickle.load(f)
        
    with open('../experimental-data/hsls/results/reduction_'+model+'_s{}_eo.pkl'.format(seed), 'rb+') as f: 
        hsls_reduction_eo = pickle.load(f)
       
    with open('../experimental-data/hsls/results/roc_'+model+'_s{}_eo.pkl'.format(seed), 'rb+') as f: 
        hsls_roc_eo = pickle.load(f)
        
    with open('../experimental-data/hsls/results/leveraging_'+model+'_s{}_eo.pkl'.format(seed), 'rb+') as f: 
        hsls_leverage = pickle.load(f)
        
        
    ## Model Projection
    with open('../experimental-data/hsls/results/hsls-mp-2022-01-25-17.12.43.pkl', 'rb+') as f: 
        hsls_mp = pickle.load(f)
        
    with open('../experimental-data/hsls/results/original_'+model+'_s{}_.pkl'.format(seed), 'rb+') as f: 
        hsls_original = pickle.load(f)
    
    if model == 'rf':
        model = 'rfc'
    hsls_tolerance = hsls_mp['tolerance']

    
    if fair == 'eo':
        return hsls_original, hsls_eqodds, hsls_reduction_eo, hsls_roc_eo, hsls_leverage, hsls_mp, hsls_tolerance
 
    else: 
        return None, hsls_reduction_sp, hsls_roc_sp, None, hsls_mp, hsls_tolerance
        
        
        
        
def plot_hsls(original, hardt, reduction, rejection, leverage, mp_all, tolerance, model, ax=None, fair='eo', mp_name='ce', alpha = 1):
    # specify number of epsilon
    idx_projection = np.arange(len(tolerance))
    # idx_projection = np.array([3, 4, 5, 6])
    idx_rejection = np.arange(len(tolerance))
    idx_reduction = np.arange(len(reduction['eps']))
    idx_calibration = []
    idx_fact= [] 

    #mp version
    data = {'original':original, 'hardt': hardt, 'reduction': reduction, 'rejection': rejection, 'leverage': leverage,  'mp_all': mp_all, 'tolerance': tolerance}
    idx = {'calibration': idx_calibration, 'reduction': idx_reduction, 'rejection': idx_rejection, 'mp': idx_projection, 'fact': idx_fact }

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.legend()
    basic_plot(data, idx, model, ax, fair, mp_name, alpha = alpha)
    
    ax.set_title(r"$\bf{HSLS}$ (" + get_model_name(model) + ')')
    if ax == None:
        fig.savefig('plot/hsls-'+model+'-' + mp_name,format='png', dpi=300)

def load_enem_plots(model, fair='eo', seed = 42):
    ## benchmarks
    with open('../experimental-data/enem/results/eqodds_'+model+'_s{}_.pkl'.format(seed), 'rb+') as f: 
        enem_eqodds = pickle.load(f)
        
    with open('../experimental-data/enem/results/reduction_'+model+'_s{}_eo.pkl'.format(seed), 'rb+') as f: 
        enem_reduction_eo = pickle.load(f)
        
    with open('../experimental-data/enem/results/roc_'+model+'_s{}_eo.pkl'.format(seed), 'rb+') as f: 
        enem_roc_eo = pickle.load(f)

        
    with open('../experimental-data/enem/results/leveraging_'+model+'_s{}_eo.pkl'.format(seed), 'rb+') as f: 
        enem_leverage = pickle.load(f)
        
    with open('../experimental-data/enem/results/original_'+model+'_s{}_.pkl'.format(seed), 'rb+') as f: 
        enem_original = pickle.load(f)
    ## Model Projection
    with open('../experimental-data/enem/results/enem-mp-2022-05-22-13.37.03.pkl', 'rb+') as f: 
        enem_mp = pickle.load(f)

    enem_tolerance = enem_mp['tolerance']
    
    
    if fair == 'eo':
        return enem_original, enem_eqodds, enem_reduction_eo, enem_roc_eo, enem_leverage, enem_mp, enem_tolerance

          
def plot_enem(original, hardt, reduction, rejection, leverage, mp_all, tolerance, model, ax=None, fair='eo', mp_name='ce', alpha = 0.5):
    
    # specify number of epsilon
    idx_projection = np.arange(len(tolerance))
    # idx_projection = np.array([3, 4, 5, 6])
    idx_rejection = np.arange(len(tolerance))
    idx_reduction = np.arange(len(tolerance))
    idx_calibration = []
    idx_fact= [] 

    #mp version
    data = {'original':original, 'hardt': hardt, 'reduction': reduction, 'rejection': rejection, 'leverage': leverage,  'mp_all': mp_all, 'tolerance': tolerance}
    idx = {'calibration': idx_calibration, 'reduction': idx_reduction, 'rejection': idx_rejection, 'mp': idx_projection, 'fact': idx_fact }

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.legend()
    basic_plot(data, idx, model, ax, fair, mp_name, alpha = alpha)

    ax.set_title(r"$\bf{ENEM}$ (" + get_model_name(model) + ')')
    if ax == None:
        fig.savefig('plot/enem-'+model+'-' + mp_name,format='png', dpi=300)



def plot_dataset(dataset, model, ax=None, fair='eo', mp_name='ce', seed = 42, alpha = 0.5): 
    if dataset == 'adult': 
        hardt, reduction, rejection, leverage, mp, original, tolerance = load_adult_plots(model, fair)
        plot_adult(hardt, reduction, rejection, leverage, mp, original, tolerance, model, ax, fair, mp_name)
    elif dataset == 'hsls': 
        original, hardt, reduction, rejection, leverage, mp, tolerance = load_hsls_plots(model, fair, seed)
        plot_hsls(original, hardt, reduction, rejection, leverage, mp, tolerance, model, ax, fair, mp_name)
    elif dataset == 'enem':
        original, hardt, reduction, rejection, leverage, mp, tolerance = load_enem_plots(model, fair, seed)
        plot_enem(original, hardt, reduction, rejection, leverage, mp, tolerance, model, ax, fair, mp_name, alpha = alpha)


def load_enem_scores(model, fair='eo', seed = 42):
    ## benchmarks
    with open('../experimental-data/enem/results/scores_eqodds_'+model+'_s{}_itr10_.pkl'.format(seed), 'rb+') as f: 
        enem_eqodds = pickle.load(f)
        
    with open('../experimental-data/enem/results/scores_reduction_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        enem_reduction_eo = pickle.load(f)
        
    with open('../experimental-data/enem/results/scores_roc_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        enem_roc_eo = pickle.load(f)
        
    with open('../experimental-data/enem/results/scores_leveraging_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        enem_leverage = pickle.load(f)
        
    with open('../experimental-data/enem/results/scores_original_'+model+'_s{}_itr10_.pkl'.format(seed), 'rb+') as f: 
        enem_original = pickle.load(f)
    ## Model Projection
    with open('../experimental-data/enem/results/enem-mp-2022-05-22-13.37.03.pkl', 'rb+') as f: 
         enem_mp = pickle.load(f)
    enem_tolerance = enem_mp['tolerance']
    return enem_original, enem_eqodds, enem_reduction_eo, enem_roc_eo, enem_leverage, enem_mp, enem_tolerance

def load_hsls_scores(model, fair='eo', seed = 42):
    ## benchmarks
    with open('../experimental-data/hsls/results/scores_eqodds_'+model+'_s{}_itr10_.pkl'.format(seed), 'rb+') as f: 
        hsls_eqodds = pickle.load(f)
        
    with open('../experimental-data/hsls/results/scores_reduction_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        hsls_reduction_eo = pickle.load(f)
        
    with open('../experimental-data/hsls/results/scores_roc_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        hsls_roc_eo = pickle.load(f)
        
    with open('../experimental-data/hsls/results/scores_leveraging_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        hsls_leverage = pickle.load(f)
        
    with open('../experimental-data/hsls/results/scores_original_'+model+'_s{}_itr10_.pkl'.format(seed), 'rb+') as f: 
        hsls_original = pickle.load(f)
    ## Model Projection
    with open('../experimental-data/hsls/results/hsls-mp-2022-01-25-17.12.43.pkl', 'rb+') as f:
        hsls_mp = pickle.load(f)
    hsls_tolerance = hsls_mp['tolerance']
    
    return hsls_original, hsls_eqodds, hsls_reduction_eo, hsls_roc_eo, hsls_leverage, hsls_mp, hsls_tolerance


def process_scores(dataset, model, fair='eo', start_seed = 33, end_seed = 42):
    # return dic {eps1:[[from itr1], [from itr2], [from itr3]], eps2:...} 
    # or list [[from itr1], [from itr2], ...]
    # [from itr_i]: num_seed x score_list
    score_original = []
    score_hardt = []
    score_reduction = []
    score_rejection = []
    score_leverage = []
    score_mp = []
    for seed in range(start_seed, end_seed+1):
        if dataset == "enem":
            original, hardt, reduction, rejection, leverage, mp, tolerance = load_enem_scores(model, fair, seed)
        elif dataset == "hsls":
            original, hardt, reduction, rejection, leverage, mp, tolerance = load_hsls_scores(model, fair, seed)
        score_original.append(original)
        score_hardt.append(hardt)
        score_reduction.append(reduction)
        score_rejection.append(rejection)
        score_leverage.append(leverage)
        score_mp.append(mp)
    # reorg score to prep for multiplicity
    score_original = process_pkl_no_eps(score_original)
    score_hardt = process_pkl_no_eps(score_hardt)
    score_leverage = process_pkl_no_eps(score_leverage)
    score_reduction = process_pkl_w_eps(score_reduction)
    score_rejection = process_pkl_w_eps(score_rejection)
    return score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp

def process_scores_per_itr(dataset, model, fair='eo', start_seed = 33, end_seed = 42):
    # return dic {eps1:[[from itr1], [from itr2], [from itr3]], eps2:...} 
    # or list [[from itr1], [from itr2], ...]
    # [from itr_i]: num_seed x score_list
    score_original = []
    score_hardt = []
    score_reduction = []
    score_rejection = []
    score_leverage = []
    score_mp = []
    for seed in range(start_seed, end_seed+1):
        if dataset == "enem":
            original, hardt, reduction, rejection, leverage, mp, tolerance = load_enem_scores(model, fair, seed)
        elif dataset == "hsls":
            original, hardt, reduction, rejection, leverage, mp, tolerance = load_hsls_scores(model, fair, seed)
        score_original.append(original)
        score_hardt.append(hardt)
        score_reduction.append(reduction)
        score_rejection.append(rejection)
        score_leverage.append(leverage)
        score_mp.append(mp)
    # reorg score to prep for multiplicity
    score_original = process_pkl_no_eps(score_original)
    score_hardt = process_pkl_no_eps(score_hardt)
    score_leverage = process_pkl_no_eps(score_leverage)
    score_reduction = pool_models_per_itr(score_reduction)
    score_rejection = pool_models_per_itr(score_rejection)

    return score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp

def process_red_original_scores_per_itr(dataset, model, fair='eo', start_seed = 33, end_seed = 42):
    # return dic {eps1:[[from itr1], [from itr2], [from itr3]], eps2:...} 
    # or list [[from itr1], [from itr2], ...]
    # [from itr_i]: num_seed x score_list
    score_original = []
    score_reduction = []        
    for seed in range(start_seed, end_seed+1):
        if dataset == "enem":
            original, reduction = load_enem_original_reduction_scores(model, fair, seed)
            score_original.append(original)
            score_reduction.append(reduction)
        elif dataset == "hsls":
            reduction  =  load_hsls_reduction_scores(model, fair, seed)
            score_reduction.append(reduction)
    if dataset=="hsls":
        for seed in range(33,43):
            original = load_hsls_original_scores(model,fair,seed)
            score_original.append(original)
    # reorg score to prep for multiplicity
    score_original = process_pkl_no_eps(score_original)
    score_reduction = pool_models_per_itr(score_reduction)
    return score_original, score_reduction

def load_enem_original_reduction_scores(model, fair = 'eo', seed = 42):
    with open('enem/benchmarks/results/scores_reduction_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        enem_reduction_eo = pickle.load(f)
    
    with open('enem/benchmarks/results/scores_original_'+model+'_s{}_itr10_.pkl'.format(seed), 'rb+') as f: 
        enem_original = pickle.load(f)
    
    return enem_original, enem_reduction_eo

def load_hsls_reduction_scores(model, fair = 'eo', seed = 42):
    with open('hsls/results/scores_reduction_'+model+'_s{}_itr10_eo.pkl'.format(seed), 'rb+') as f: 
        hsls_reduction_eo = pickle.load(f)

def load_hsls_original_scores(model, fair = 'eo', seed = 42):
    
    with open('hsls/results/scores_original_'+model+'_s{}_itr10_.pkl'.format(seed), 'rb+') as f: 
        hsls_original = pickle.load(f)
    
    return hsls_original

def plot_dataset_multiplicity(dataset, model, ax=None, fair='eo', mp_name='ce', 
                                    start_seed = 33, end_seed = 42, alpha = 0.5, quantile = .8): 
    
    # combine eo_mean, eo_sd
    if fair=='eo_ambiguity' or fair =="eo_score_std":
        fair_plot = fair
        fair = "eo"
    if dataset == "enem":
        original, hardt, reduction, rejection, leverage, mp, tolerance = load_enem_plots(model, fair, start_seed)
    elif dataset == "hsls":
        original, hardt, reduction, rejection, leverage, mp, tolerance = load_hsls_plots(model, fair, start_seed)
    leverage["meo_std"] = [np.std(leverage["meo"], ddof = 1)]
# =============================================================================
#     for seed in range(start_seed+1, end_seed+1):
#         original_temp, hardt_temp, reduction_temp, rejection_temp, leverage_temp, mp_temp, _ = load_enem_plots(model, fair, seed)
#         original["abseo_mean"].append(original_temp["abseo_mean"])
#         hardt["abseo_mean"].append(hardt_temp["abseo_mean"])
#         leverage["meo"] = np.concatenate([leverage["meo"], leverage_temp["meo"]])
#         
#         reduction["abseo_mean"] += reduction_temp["abseo_mean"]
#         rejection["abseo_mean"] += rejection_temp["abseo_mean"]
#         
# 
#         original["abseo_std"].append(original_temp["abseo_std"])
#         hardt["abseo_std"].append(hardt_temp["abseo_std"])
#         leverage["meo_std"].append(np.std(leverage_temp["meo"], ddof = 1))
#         
#         reduction["abseo_std"] += reduction_temp["abseo_std"]
#         rejection["abseo_std"] += rejection_temp["abseo_std"]
#         
#     num_seed = len(original["abseo_mean"])
#     original["abseo_mean"] = np.average(original["abseo_mean"]) 
#     hardt["abseo_mean"] = np.average(hardt["abseo_mean"]) 
#     leverage["meo"] = np.average(leverage["meo"]) 
#     reduction["abseo_mean"] = [x/num_seed for x in reduction["abseo_mean"]] 
#     rejection["abseo_mean"] = [x/num_seed for x in rejection["abseo_mean"]] 
#     
#     original["abseo_std"] = np.average(original["abseo_std"]) 
#     hardt["abseo_std"] = np.average(hardt["abseo_std"]) 
#     leverage["meo_std"] = np.average(leverage["meo_std"]) 
#     reduction["abseo_std"] = [x/num_seed for x in reduction["abseo_std"]] 
#     rejection["abseo_std"] = [x/num_seed for x in rejection["abseo_std"]] 
# =============================================================================
    # comment out the following if the above uncommented.
    leverage["meo"] = np.average(leverage["meo"]) 
    
    score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp = process_scores(dataset, model, fair, start_seed, end_seed)
    ambiguity_original = compute_ambiguity_list(score_original)
    ambiguity_hardt = compute_ambiguity_list(score_hardt)
    ambiguity_reduction = compute_ambiguity_dic(score_reduction)
    ambiguity_rejection = compute_ambiguity_dic(score_rejection)
    ambiguity_leverage = compute_ambiguity_list(score_leverage)

    # ambiguity_mp = compute_ambiguity_dic(score_mp)
    scoresd_original = compute_score_sd_list(score_original, quantile)
    scoresd_hardt = compute_score_sd_list(score_hardt, quantile)
    scoresd_reduction = compute_score_sd_dic(score_reduction, quantile)
    scoresd_rejection = compute_score_sd_dic(score_rejection, quantile)
    scoresd_leverage = compute_score_sd_list(score_leverage, quantile)
    
    
    # add ambiguityto above list 
    original["ambiguity_mean"] = np.average(ambiguity_original)
    hardt["ambiguity_mean"] = np.average(ambiguity_hardt)
    leverage["ambiguity_mean"] = np.average(ambiguity_leverage) 
    reduction["ambiguity_mean"] = [np.average(amb_eps) for amb_eps in ambiguity_reduction.values()]
    rejection["ambiguity_mean"] = [np.average(amb_eps) for amb_eps in ambiguity_rejection.values()]    
    
  
    original["ambiguity_std"] = np.std(ambiguity_original,  ddof=1)
    hardt["ambiguity_std"] = np.std(ambiguity_hardt,  ddof=1)
    leverage["ambiguity_std"] = np.std(ambiguity_leverage, ddof=1)
    reduction["ambiguity_std"] = [np.std(amb_eps, ddof=1) for amb_eps in ambiguity_reduction.values()]
    rejection["ambiguity_std"] = [np.std(amb_eps, ddof=1) for amb_eps in ambiguity_rejection.values()]
    
    # add scores_std to above list
    original["score_std_mean"] = np.average(scoresd_original)
    hardt["score_std_mean"] = np.average(scoresd_hardt)
    leverage["score_std_mean"] = np.average(scoresd_leverage) 
    reduction["score_std_mean"] = [np.average(sc_eps) for sc_eps in scoresd_reduction.values()]
    rejection["score_std_mean"] = [np.average(sc_eps) for sc_eps in scoresd_rejection.values()]    
    
    original["score_std_std"] = np.std(scoresd_original,  ddof=1)
    hardt["score_std_std"] = np.std(scoresd_hardt,  ddof=1)
    leverage["score_std_std"] = np.std(scoresd_leverage, ddof=1)
    reduction["score_std_std"] = [np.std(sc_eps, ddof=1) for sc_eps in scoresd_reduction.values()]
    rejection["score_std_std"] = [np.std(sc_eps, ddof=1) for sc_eps in scoresd_rejection.values()]
    
    # initialize for mp, to be changed 
    mp["ambiguity_mean"] = 0
    mp["ambiguity_std"] = 0
    mp["score_std_mean"] = 0
    mp["score_std_std"] = 0
    # plot
    if dataset == "enem":
        plot_enem(original, hardt, reduction, rejection, leverage, mp, tolerance, model, ax, fair_plot, mp_name, alpha = alpha)
    elif dataset == "hsls":
        plot_hsls(original, hardt, reduction, rejection, leverage, mp, tolerance, model, ax, fair_plot, mp_name, alpha = alpha)
        
def plot_dataset_diff_multiplicity(dataset, model, ax=None, fair='eo', mp_name='ce', 
                                    start_seed = 33, end_seed = 42, alpha = 0.5, quantile = 0.8): 
    
    # combine eo_mean, eo_sd
    if fair=='eo_ambiguity' or fair =="eo_score_std":
        fair_plot = fair
        fair = "eo"
    if dataset == "enem":
        original, hardt, reduction, rejection, leverage, mp, tolerance = load_enem_plots(model, fair, start_seed)
    elif dataset == "hsls":
        original, hardt, reduction, rejection, leverage, mp, tolerance = load_hsls_plots(model, fair, start_seed)
    leverage["meo_std"] = [np.std(leverage["meo"], ddof = 1)]
# =============================================================================
#     for seed in range(start_seed+1, end_seed+1):
#         original_temp, hardt_temp, reduction_temp, rejection_temp, leverage_temp, mp_temp, _ = load_enem_plots(model, fair, seed)
#         original["abseo_mean"].append(original_temp["abseo_mean"])
#         hardt["abseo_mean"].append(hardt_temp["abseo_mean"])
#         leverage["meo"] = np.concatenate([leverage["meo"], leverage_temp["meo"]])
#         
#         reduction["abseo_mean"] += reduction_temp["abseo_mean"]
#         rejection["abseo_mean"] += rejection_temp["abseo_mean"]
#        
# 
#         original["abseo_std"].append(original_temp["abseo_std"])
#         hardt["abseo_std"].append(hardt_temp["abseo_std"])
#         leverage["meo_std"].append(np.std(leverage_temp["meo"], ddof = 1))
#         
#         reduction["abseo_std"] += reduction_temp["abseo_std"]
#         rejection["abseo_std"] += rejection_temp["abseo_std"]
#         
#     num_seed = len(original["abseo_mean"])
#     original["abseo_mean"] = np.average(original["abseo_mean"]) 
#     hardt["abseo_mean"] = np.average(hardt["abseo_mean"]) 
#     leverage["meo"] = np.average(leverage["meo"]) 
#     reduction["abseo_mean"] = [x/num_seed for x in reduction["abseo_mean"]] 
#     rejection["abseo_mean"] = [x/num_seed for x in rejection["abseo_mean"]] 
#     
#     original["abseo_std"] = np.average(original["abseo_std"]) 
#     hardt["abseo_std"] = np.average(hardt["abseo_std"]) 
#     leverage["meo_std"] = np.average(leverage["meo_std"]) 
#     reduction["abseo_std"] = [x/num_seed for x in reduction["abseo_std"]] 
#     rejection["abseo_std"] = [x/num_seed for x in rejection["abseo_std"]] 
# =============================================================================
    # comment out the following if the above uncommented.
    leverage["meo"] = np.average(leverage["meo"]) 
    
    score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp = process_scores(dataset, model, fair, start_seed, end_seed)
    ambiguity_original, scoresd_original = compute_diff_multiplicity_list(score_original, quantile)
    ambiguity_hardt, scoresd_hardt = compute_diff_multiplicity_list(score_hardt, quantile)
    ambiguity_reduction, scoresd_reduction = compute_diff_multiplicity_dic(score_reduction, quantile)
    ambiguity_rejection, scoresd_rejection = compute_diff_multiplicity_dic(score_rejection, quantile)
    ambiguity_leverage, scoresd_leverage = compute_diff_multiplicity_list(score_leverage, quantile)

    
    # add diff_ambiguity to above list 
    original["ambiguity_mean"] = np.average(ambiguity_original)
    hardt["ambiguity_mean"] = np.average(ambiguity_hardt)
    leverage["ambiguity_mean"] = np.average(ambiguity_leverage) 
    reduction["ambiguity_mean"] = [np.average(amb_eps) for amb_eps in ambiguity_reduction.values()]
    rejection["ambiguity_mean"] = [np.average(amb_eps) for amb_eps in ambiguity_rejection.values()]    
    
  
    original["ambiguity_std"] = np.std(ambiguity_original,  ddof=1)
    hardt["ambiguity_std"] = np.std(ambiguity_hardt,  ddof=1)
    leverage["ambiguity_std"] = np.std(ambiguity_leverage, ddof=1)
    reduction["ambiguity_std"] = [np.std(amb_eps, ddof=1) for amb_eps in ambiguity_reduction.values()]
    rejection["ambiguity_std"] = [np.std(amb_eps, ddof=1) for amb_eps in ambiguity_rejection.values()]
    
    # add scores_std to above list
    original["score_std_mean"] = np.average(scoresd_original)
    hardt["score_std_mean"] = np.average(scoresd_hardt)
    leverage["score_std_mean"] = np.average(scoresd_leverage) 
    reduction["score_std_mean"] = [np.average(sc_eps) for sc_eps in scoresd_reduction.values()]
    rejection["score_std_mean"] = [np.average(sc_eps) for sc_eps in scoresd_rejection.values()]    
    
    original["score_std_std"] = np.std(scoresd_original,  ddof=1)
    hardt["score_std_std"] = np.std(scoresd_hardt,  ddof=1)
    leverage["score_std_std"] = np.std(scoresd_leverage, ddof=1)
    reduction["score_std_std"] = [np.std(sc_eps, ddof=1) for sc_eps in scoresd_reduction.values()]
    rejection["score_std_std"] = [np.std(sc_eps, ddof=1) for sc_eps in scoresd_rejection.values()]
    
    # initialize for mp, to be changed 
    mp["ambiguity_mean"] = 0
    mp["ambiguity_std"] = 0
    mp["score_std_mean"] = 0
    mp["score_std_std"] = 0
    # plot
    if dataset == "enem":
        plot_enem(original, hardt, reduction, rejection, leverage, mp, tolerance, model, ax, fair_plot, mp_name, alpha = alpha)
    elif dataset == "hsls":
        plot_hsls(original, hardt, reduction, rejection, leverage, mp, tolerance, model, ax, fair_plot, mp_name, alpha = alpha)
