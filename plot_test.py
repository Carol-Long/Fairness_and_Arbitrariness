#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:42:16 2023

@author: carollong
"""

from plot_figures_integrated import plot_dataset
import matplotlib.pyplot as plt
# choice hsls
#dataset = 'enem'
dataset = 'enem'
fair = 'eo'
mp_name = 'ce' 

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
for seed in range(33,43):
    plot_dataset(dataset, 'rf', ax, fair, mp_name, seed, alpha = 0.6)
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# plot_dataset(dataset, 'logit', axes[0], fair, mp_name)
# plot_dataset(dataset, 'rf', axes[1], fair, mp_name)

# plot_dataset(dataset, 'gbm', axes[2], fair, mp_name)
lines = ax.get_lines()
plt.legend(handles = lines[:6], labels = ["EqOdds","Rejection","Reduction","LevEqOpp","Fair Projection","Base"], 
           bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig(dataset+'-'+fair+'-'+mp_name + "-33-42" +'.pdf', dpi=300)


