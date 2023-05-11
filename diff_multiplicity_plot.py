#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:41:01 2023

@author: carollong
"""

from plot_figures_integrated import plot_dataset_diff_multiplicity
import matplotlib.pyplot as plt
dataset = 'enem'
fair = 'eo_ambiguity'
mp_name = 'ce'
fig, ax = plt.subplots(1, 1, figsize=(5, 3)) 
plot_dataset_diff_multiplicity(dataset, 'rf', ax, fair, mp_name, start_seed = 33, end_seed = 42, alpha = 0.8, quantile = .8)
ax.set_ylabel('Mean Delta Ambiguity')
lines = ax.get_lines()
plt.legend(handles = lines[:5], labels = ["EqOdds","Rejection","Reduction","LevEqOpp","Base"])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig(dataset + "-eo-delta_ambiguity" +'.pdf', dpi=300)


fair = 'eo_score_std'
fig, ax = plt.subplots(1, 1, figsize=(5, 3)) 
plot_dataset_diff_multiplicity(dataset, 'rf', ax, fair, mp_name, start_seed = 33, end_seed = 42, alpha = 0.8, quantile = .8)
ax.set_ylabel('Mean Delta Score Std')
lines = ax.get_lines()
plt.legend(handles = lines[:5], labels = ["EqOdds","Rejection","Reduction","LevEqOpp","Base"])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#plt.legend(handles = lines[:4], labels = ["EqOdds","Rejection","Reduction","Base"], 
           #bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(dataset + "-eo-delta_score_std" +'.pdf', dpi=300)
