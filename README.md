# TPOT Documentation
## Introduction
TPOT is an AutoML library that creates an optimized pipeline for the task of  classification or regression. It constitutes of grid search over various models with certain hyperparameters. The attributes *generations*, *population_size* and *cv*, which controls the total numbers of model fit to the task and optimized model is returned.

$$
models = \mathrm{generations} \times \mathrm{population size} \times \mathrm{cv}
$$

Further, *cv* represents cross-validations i.e. TPOT is based on average scoring of *cross_val_scores*. Hence it is built over libraries like **scikit-learn**, **scipy**, **PyTorch**, etc.

## Creating Optimized Pipeline
