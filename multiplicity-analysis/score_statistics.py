#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 05:02:47 2023
Plot on Per-Sample Mean scores in Competing Models
@author: carollong
"""
from multiplicity_helper import *
from plot_figures_integrated import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# option: hsls/enem/adult
data = "hsls"

fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
# option: logit, rf, gbm
model_base = 'rf'
score_original, score_hardt, score_reduction, score_rejection, score_leverage, score_mp = process_scores_per_itr(data, model_base, fair='eo', start_seed = 33, end_seed = 42)
# threshold baseline
threshold = 0.5
score_threshold_original = [[np.where(scores >= threshold, 1, 0) for scores in sublist] for sublist in score_original]


mean_scores = (pd.DataFrame(np.squeeze(score_original[0])).mean()).hist(bins=bins,alpha=0.5)
#mean_scores_thresholded_original = (pd.DataFrame(np.squeeze(score_threshold_original[2])).mean()).hist(bins=bins,alpha=0.5)
mean_scores_red = (pd.DataFrame(np.squeeze(score_reduction[0])).mean()).hist(bins=bins,alpha=0.5)
#mean_scores_hardt = (pd.DataFrame(np.squeeze(score_hardt[1])).mean()).hist(bins=bins,alpha=0.5)
mean_scores_rejection = (pd.DataFrame(np.squeeze(score_rejection[0])).mean()).hist(bins=bins,alpha=0.5)
#mean_scores_leverage = (pd.DataFrame(np.squeeze(score_leverage[0])).mean()).hist(bins=bins,alpha=0.5)
# =============================================================================
# mean_scores = (pd.DataFrame(np.squeeze(score_original[0])).mean()).quantile(quantiles)
# mean_scores_thresholded_original = (pd.DataFrame(np.squeeze(score_threshold_original[0])).mean()).quantile(quantiles)
# 
# mean_scores_red = (pd.DataFrame(np.squeeze(score_reduction[0])).mean()).quantile(quantiles)
# mean_scores_hardt = (pd.DataFrame(np.squeeze(score_hardt[0])).mean()).quantile(quantiles)
# mean_scores_rejection = (pd.DataFrame(np.squeeze(score_rejection[0])).mean()).quantile(quantiles)
# mean_scores_leverage = (pd.DataFrame(np.squeeze(score_leverage[0])).mean()).quantile(quantiles)
# =============================================================================
#plt.legend(["Baseline","Thresholded Baseline","Reduction","EqOddds","Hardt","Rejection","Leverage"])
plt.legend(["Baseline","Reduction","Rejection"])
plt.show()

# =============================================================================
# plt.plot(quantiles, mean_scores, label='Baseline', marker='o',markersize = 0.2)
# plt.plot(quantiles, mean_scores_thresholded_original, label='Thresholded Baseline', marker='o',markersize = 0.2)
# 
# plt.plot(quantiles, mean_scores_red, label='Reduction', marker='o',markersize = 0.2)
# plt.plot(quantiles, mean_scores_hardt, label='EqOdds', marker='o',markersize = 0.2)
# plt.plot(quantiles, mean_scores_rejection, label='Rejection', marker='o',markersize = 0.2)
# plt.plot(quantiles, mean_scores_leverage, label='Leverage', marker='o',markersize = 0.2)
# =============================================================================
#plt.plot(quantiles, mean_scores_mp, label='MP', marker='o',markersize = 1)

# =============================================================================
# plt.ylabel('Quantiles')
# plt.xlabel('Per-sample Mean scores')
# #plt.legend(loc='lower right')
# plt.legend()
# # option: hsls/enem/adult
# plt.title('Mean scores in Competing Models (HSLS)')
# plt.tight_layout()
# # option: hsls/enem/adult
# plt.savefig('Mean_scores_HSLS.pdf', dpi=300)
# =============================================================================
