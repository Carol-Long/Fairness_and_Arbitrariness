# Fairness_and_Arbitrariness
The official code of **Arbitrariness Lies Beyond the Fairness-Accuracy Frontier** (NeurIPS 2023 Spotlight) [[arXiv]](https://arxiv.org/pdf/2306.09425.pdf)

## Abstract
Machine learning tasks may admit multiple competing models that achieve similar performance yet produce conflicting outputs for individual samples--- a phenomenon known as **predictive multiplicity**. We demonstrate that fairness interventions in machine learning optimized solely for group fairness and accuracy can **exacerbate** predictive multiplicity. Consequently, state-of-the-art fairness interventions can mask high predictive multiplicity behind favorable group fairness and accuracy metrics. We argue that a third axis of "arbitrariness" should be considered when deploying models to aid decision-making in applications of individual-level impact. To address this challenge, we propose an ensemble algorithm applicable to any fairness intervention that provably ensures more consistent predictions.

## Training Baseline and Fair Models 
(adapted from [FairProjection [3]](https://github.com/HsiangHsu/Fair-Projection#data-contains-all-datasets))
### Environmental Setup
```
conda create --n fair_arbitrariness python=3.8
conda activate fair_arbitrariness
pip install aif360 inFairness skorch IPython
pip install numpy --upgrade
```

### Dataset
- `UCI-Adult/`: raw data <ins> adult.data</ins> , <ins> adult.names</ins> , <ins> adult.test</ins>  [1].
- `HSLS/`: k-NN imputed HSLS dataset [2] (Raw data and pre-processing: https://drive.google.com/drive/folders/14Ke1fiB5RKOVlA8iU9aarAeJF0g4SdBl).
- `ENEM/`: downsampled pre-processed data from https://download.inep.gov.br/microdados/microdados_enem_2020.zip [3].

### Command
```python3 run_benchmark.py -m [baseline model] -f [fairness constraint] -n [num_itr] -i [input dataset] -s [randome seed] &```\
Options for arguments:
- [baseline model]: gbm, logit, rf (Default: gbm)
- [fair method]: eqodds, leveraging, reduction, roc, original (Default: reduction)
- [constraint]: eo, sp, (Default: eo)
- [num iter]: Any positive integer (Default: 10)
- [input dataset]: hsls, enem, adult, german, compas (Default: enem)
- [random seed]: Any integer (Default: 42)
e.g.,
```
  for i in {33..42}
  do
  python3 run_benchmark.py -m rf -f reduction -n 10 -i enem -s $i
  done
```
Fairness methods include EqOdds [4], LevEqOpp [5], Reduction [6], and Rejection [7]. Original means no fairness intervention is applied.

#### `experimental-results/`: stores all baseline and fair model scores and predictions as pickle files.

## Arbitrariness (multiplicity) analysis
#### `multiplicity-analysis/all_plots.ipynb`: contains reproducible figures for the paper.


## Reference
[1] M. Lichman. UCI machine learning repository, 2013.

[2] Ingels, S. J., Pratt, D. J., Herget, D. R., Burns, L. J., Dever, J. A., Ottem, R., Rogers, J. E., Jin, Y., and Leinwand, S. (2011). High school longitudinal study of 2009 (hsls: 09): Base-year data file documentation. nces 2011-328. National Center for Education Statistics.

[3] Alghamdi, Wael, Hsiang Hsu, Haewon Jeong, Hao Wang, Peter Michalak, Shahab Asoodeh, and Flavio Calmon. Beyond Adult and COMPAS: Fair multi-class prediction via information projection. Advances in Neural Information Processing Systems 35:38747-38760, 2022.

[4] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 29:3315–3323, 2016.

[5] Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, Luca Oneto, and Massimiliano Pontil. Leveraging labeled and unlabeled data for consistent fair binary classification. Advances in Neural Information Processing Systems, 32, 2019.

[6] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. A reductions approach to fair classification. In International Conference on Machine Learning, pages 60–69. PMLR, 2018.

[7] F. Kamiran, A. Karim, and X. Zhang. Decision theory for discrimination-aware classification. In 2012 IEEE 12th International Conference on Data Mining, pages 924–929, Dec 2012.

