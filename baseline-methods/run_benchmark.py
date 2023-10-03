#!/usr/bin/python

import sys
import os
import getopt
import pickle
import pandas as pd

from benchmark import Benchmark
from DataLoader import *

sys.path.append('../data/HSLS')
from hsls_utils import *

sys.path.append('../leveraging-python')
from leveraging.utils import leveraging_approach
from aif360.datasets import AdultDataset, GermanDataset

import warnings
warnings.filterwarnings("ignore")


def main(argv):
    model = 'gbm'
    fair = 'reduction'
    seed = 42
    constraint = 'eo'
    num_iter = 10
    inputfile = 'enem'
    
    try:
        opts, args = getopt.getopt(argv,"hm:s:f:c:n:i:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Correct Usage:\n')
            print('python run_benchmark.py -m [model name] -f [fair method] -c [constraint] -n [num iter] -i [inputfile] -s [seed]')
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
        
        
    path = '../data/'
    if inputfile == 'hsls':
        file = 'HSLS/hsls_knn_impute.pkl'
        df = load_hsls_imputed(path, file, [])
        privileged_groups = [{'racebin': 1}]
        unprivileged_groups = [{'racebin': 0}]
        protected_attrs = ['racebin']
        label_name = 'gradebin'
        
    elif inputfile == 'enem':
        df = pd.read_pickle(path+'ENEM/enem-50000-20.pkl')
        privileged_groups = [{'racebin': 1}]
        unprivileged_groups = [{'racebin': 0}]
        protected_attrs = ['racebin']
        label_name = 'gradebin'
        df[label_name] = df[label_name].astype(int)
        
    elif inputfile == 'adult':
        df = load_data('adult')
        privileged_groups = [{'gender': 1}]
        unprivileged_groups = [{'gender': 0}]
        protected_attrs = ['gender']
        label_name = 'income'
    
    elif inputfile == 'german':
        df = GermanDataset().convert_to_dataframe()[0]
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        protected_attrs = ['sex']
        label_name = 'credit'

    elif inputfile == 'compas':
        df = load_data('compas')
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        protected_attrs = ['race']
        label_name = 'is_recid'
        
    else: 
        print('Invalid Input Dataset Name')
        sys.exit(2)

    print('#### Data Loaded. ')
    

    #### Setting group attribute and label ####
    
    
    bm = Benchmark(df, privileged_groups, unprivileged_groups, protected_attrs,label_name)


    #### Run benchmarks ####
    if fair == 'reduction':
        eps_list = [0.1, 1, 4, 6, 8, 9, 10, 12] # enem
        if constraint == 'sp':
            results, compiled_scores_list, compiled_scores_valid_list = bm.reduction(model, num_iter, seed, params=eps_list, constraint='DemographicParity')
        elif constraint == 'eo':
            results, compiled_scores_list, compiled_scores_valid_list = bm.reduction(model, num_iter, seed, params=eps_list, constraint='EqualizedOdds')
        
    elif fair == 'eqodds':
        results, compiled_scores_list, compiled_scores_valid_list = bm.eqodds(model, num_iter, seed)
        constraint = ''
        
    elif fair == 'roc':
        eps_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
        if constraint == 'sp':
            results, compiled_scores_list, compiled_scores_valid_list  = bm.roc(model, num_iter, seed, params=eps_list, constraint='DemographicParity')
        elif constraint == 'eo':
            results, compiled_scores_list, compiled_scores_valid_list = bm.roc(model, num_iter, seed, params=eps_list, constraint='EqualizedOdds')
        
    elif fair == 'leveraging':
        _, results, _, compiled_scores_list, compiled_scores_valid_list = leveraging_approach(df, protected_attrs, label_name, use_protected=True, model = model, num_iter = num_iter, rand_seed =seed)
        
    elif fair == 'original':
        results, compiled_scores_list, compiled_scores_valid_list = bm.original(model, num_iter, seed)
        constraint = ''
        
    else:
        print('Undefined method')
        sys.exit(2)


    result_path = './{}/results/'.format(inputfile)
    filename = fair+'_'+model+'_s'+str(seed)+'_' + constraint+'.pkl'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path+filename, 'wb+') as f: 
        pickle.dump(results, f)

    score_filename = "scores_"+ fair+'_'+model+'_s'+str(seed)+'_itr' + str(num_iter) + "_" + constraint+'.pkl'

    with open(result_path+score_filename, 'wb+') as f: 
        pickle.dump(compiled_scores_list, f)    

    valid_score_filename = "valid_scores_"+ fair+'_'+model+'_s'+str(seed)+'_itr' + str(num_iter) + "_" + constraint+'.pkl'

    with open(result_path+valid_score_filename, 'wb+') as f: 
        pickle.dump(compiled_scores_valid_list, f)

if __name__ == "__main__":
    main(sys.argv[1:])
