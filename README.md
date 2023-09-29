# Fairness_and_Arbitrariness
The official code of **Arbitrariness Lies Beyond the Fairness-Accuracy Frontier** **(NeurIPS 2023 Spotlight)** [arXiv](https://arxiv.org/pdf/2306.09425.pdf)

## Abstract
Machine learning tasks may admit multiple competing models that achieve similar performance yet produce conflicting outputs for individual samples---a phenomenon known as predictive multiplicity. We demonstrate that fairness interventions in machine learning optimized solely for group fairness and accuracy can exacerbate predictive multiplicity. Consequently, state-of-the-art fairness interventions can mask high predictive multiplicity behind favorable group fairness and accuracy metrics. We argue that a third axis of ``arbitrariness'' should be considered when deploying models to aid decision-making in applications of individual-level impact. To address this challenge, we propose an ensemble algorithm applicable to any fairness intervention that provably ensures more consistent predictions.

## Reproduce Results
### Environment
conda create --n fair_arbitrariness python=3.8
conda activate fair_arbitrariness
pip install aif360 inFairness skorch IPython
pip install numpy --upgrade
### Baseline and Fair Model
Baseline codes adapted from [FairProjection](https://github.com/HsiangHsu/Fair-Projection#data-contains-all-datasets).
`python3 run_benchmark.py -m [baseline model] -f [fairness constraint] -n [num_itr] -i [input dataset] -s [randome seed] &`
Options for arguments:
[baseline model]: gbm, logit, rf (Default: gbm)
[fair method]: reduction, eqodds, roc, leveraging, original (Default: reduction)
[constraint]: eo, sp, (Default: eo)
[num iter]: Any positive integer (Default: 10)
[input dataset]: hsls, enem, adult, german, compas (Default: enem)
[random seed]: Any integer (Default: 42)
For example,
```
  for i in {33..42}
  do
  python3 run_benchmark.py -m rf -f reduction -n 10 -i enem -s $i
  done
```
### Visualize Models

### Arbitrariness (multiplicity) analysis

