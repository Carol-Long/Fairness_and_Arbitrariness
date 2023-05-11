#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 06:49:35 2023

@author: carollong
"""

from plot_figures_integrated import plot_dataset_multiplicity
import matplotlib.pyplot as plt
dataset = 'enem'
fair = 'eo_ambiguity'
mp_name = 'ce'
fig, ax = plt.subplots(1, 1, figsize=(5, 3)) 
plot_dataset_multiplicity(dataset, 'rf', ax, fair, mp_name, start_seed = 33, end_seed = 42, alpha = 0.8)
lines = ax.get_lines()
plt.legend(handles = lines[:5], labels = ["EqOdds","Rejection","Reduction","LevEqOpp","Base"], 
           bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(dataset+'-'+fair+'-'+mp_name + "-ambiguity" +'.pdf', dpi=300)


fair = 'eo_score_std'
fig, ax = plt.subplots(1, 1, figsize=(5, 3)) 

plot_dataset_multiplicity(dataset, 'rf', ax, fair, mp_name, start_seed = 33, end_seed = 42, alpha = 0.8)
lines = ax.get_lines()
plt.legend(handles = lines[:5], labels = ["EqOdds","Rejection","Reduction","LevEqOpp","Base"], 
           bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(dataset+'-'+fair+'-'+mp_name + "-score_std" +'.pdf', dpi=300)
