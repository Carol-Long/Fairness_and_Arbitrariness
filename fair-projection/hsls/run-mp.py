## standard packages
import sys
import os
import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm
from time import localtime, strftime
import time
import getopt

## scikit learn
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

## aif360
from aif360.datasets import StandardDataset

## custom packages
from utils import load_hsls_imputed, MP_tol

def main(argv):
    fair = "mp"
    model = "rf"
    seed = 42
    num_iter = 3
    constraint = "eo"
    inputfile = "hsls"

    try:
        opts, args = getopt.getopt(argv,"hm:s:f:c:n:i:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Correct Usage:\n')
            print('python run_mp.py -m [model name] -f [fair method] -c [constraint] -n [num iter] -i [inputfile] -s [seed]')
            print('\n')
            print('Options for arguments:')
            print('[model name]: gbm, logit, rf (Default: gbm)')
            print('[fair method]: reduction, eqodds, roc (Default: reduction)')
            print('[constraint]: eo, sp, (Default: eo)')
            print('[num iter]: Any positive integer (Default: 10) ')
            print('[inputfile]: hsls, enem-20000, enem-50000, ...  (Default: hsls)')
            print('[seed]: Any integer (Default: 42)')
            print('\n')
            sys.exit()
        elif opt == "-m":
            model = arg
        elif opt  == "-s":
            seed = int(arg)
        elif opt == '-f':
            fair = arg
        elif opt == '-c':
            constraint = arg
        elif opt == '-n':
            num_iter = int(arg)
        elif opt == '-i':
            inputfile = arg

    ## load HSLS dataset
    all_vars = []
    hsls_path = '../../data/HSLS/'
    hsls_file = 'hsls_knn_impute.pkl'
    df = load_hsls_imputed(hsls_path, hsls_file, all_vars)
    use_protected = True
    use_sample_weight = True
    tune_threshold = False

    tolerance = [0.000, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    start_time = time.localtime()
    start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
    filename = 'hsls-'+ str(df.shape[0]) +'-mp-' + start_time_str
    f = open(filename+'-log.txt','w')

    f.write('Setup Summary\n')
    f.write(' Sampled Dataset Shape: ' + str(df.shape) + '\n')
    f.write(' num_iter: '+str(num_iter) + '\n')
    f.write(' use_protected: '+str(use_protected) + '\n')
    f.write(' use_sample_weight: '+str(use_sample_weight) + '\n')
    f.write(' tune_threshold: '+str(tune_threshold) + '\n')
    f.write(' tolerance: '+str(tolerance) + '\n')
    f.flush()

    # ## GBM
    # f.write('GMB - CE - meo\n')
    # gbm_ce_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='cross-entropy', num_iter=num_iter, rand_seed=seed, constraint='meo')

    # ##
    # f.write('GMB - KL - meo\n')
    # gbm_kl_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=num_iter, rand_seed=seed, constraint='meo')
    # #
    # # ##
    # f.write('GMB - CE - sp\n')
    # gbm_ce_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='cross-entropy', num_iter=num_iter, rand_seed=seed, constraint='sp')
    # #
    # # ##
    # f.write('GMB - KL - sp\n')
    # gbm_kl_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=num_iter, rand_seed=seed, constraint='sp')
    # #
    # # ## Logit
    # f.write('Logit - CE - meo\n')
    # logit_ce_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='cross-entropy', num_iter=num_iter, rand_seed=seed, constraint='meo')
    # #
    # # ##
    # f.write('Logit - KL - meo\n')
    # logit_kl_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='kl', num_iter=num_iter, rand_seed=seed, constraint='meo')
    # #
    # # ##
    # f.write('Logit - CE - sp\n')
    # logit_ce_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='cross-entropy', num_iter=num_iter, rand_seed=seed, constraint='sp')
    # #
    # # ##
    # f.write('Logit - KL - sp\n')
    # logit_kl_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='kl', num_iter=num_iter, rand_seed=seed, constraint='sp')
    # #
    # ## Random Forest
    f.write('RFC - CE - meo\n')
    rfc_ce_meo, compiled_scores_list, compiled_scores_valid_list = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=num_iter, rand_seed=seed, constraint='meo')

    result_path = '../../baseline-methods/results/'
    score_filename = "scores_"+ fair+'_'+model+'_s'+str(seed)+'_itr' + str(num_iter) + "_" + constraint+'.pkl'
    if not os.path.exists(result_path):
            os.makedirs(result_path)

    with open(result_path+ score_filename, 'wb+') as pickle_f:
        pickle.dump(compiled_scores_list, pickle_f, 2)

    valid_score_filename = "valid_scores_"+ fair+'_'+model+'_s'+str(seed)+'_itr' + str(num_iter) + "_" + constraint+'.pkl'
    with open(result_path+ valid_score_filename, 'wb+') as pickle_f:
        pickle.dump(compiled_scores_valid_list, pickle_f, 2)

    #
    # # ##
    # f.write('RFC - KL - meo\n')
    # rfc_kl_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='kl', num_iter=num_iter, rand_seed=seed, constraint='meo')
    # #
    # # ##
    # f.write('RFC - CE - sp\n')
    # rfc_ce_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=num_iter, rand_seed=seed, constraint='sp')
    # #
    # # ##
    # f.write('RFC - KL - sp\n')
    # rfc_kl_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='kl', num_iter=num_iter, rand_seed=seed, constraint='sp')


    save = {
        # 'gbm_ce_meo': gbm_ce_meo,
        # 'gbm_kl_meo': gbm_kl_meo,
        # 'gbm_ce_sp': gbm_ce_sp,
        # 'gbm_kl_sp': gbm_kl_sp,
        # 'logit_ce_meo': logit_ce_meo,
        # 'logit_kl_meo': logit_kl_meo,
        # 'logit_ce_sp': logit_ce_sp,
        # 'logit_kl_sp': logit_kl_sp,
        'rfc_ce_meo': rfc_ce_meo,
        # 'rfc_kl_meo': rfc_kl_meo,
        # 'rfc_ce_sp': rfc_ce_sp,
        # 'rfc_kl_sp': rfc_kl_sp,
        'tolerance': tolerance
    }

    savename = 'hsls-mp-'+start_time_str+'.pkl'
    with open(savename, 'wb+') as pickle_f:
        pickle.dump(save, pickle_f, 2)

    f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
    f.write('Finished!!!\n')
    f.flush()
    f.close()

if __name__ == "__main__":
    main(sys.argv[1:])
