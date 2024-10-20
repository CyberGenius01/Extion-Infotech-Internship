
# TPOT Documentation

## Introduction

TPOT is an **AutoML library** designed to automate the process of building optimized machine learning pipelines for **classification** or **regression tasks**. It performs a **grid search** over various models with different hyperparameters, identifying the best-performing pipeline based on cross-validation scores. Key parameters controlling the optimization process include:

- **generations**: Number of iterations to improve the pipeline.
- **population_size**: Number of pipelines evaluated in each generation.
- **cv**: Number of cross-validations to calculate the average model performance.

The total number of models TPOT fits is given by:

$$
\text{models} = \text{generations} \times \text{population size} \times \text{cv}
$$

Since TPOT relies on cross-validation to ensure generalization, it uses the average score from `cross_val_score` during evaluation. It leverages underlying libraries like **scikit-learn**, **SciPy**, and **PyTorch** to build pipelines efficiently.

---

## Creating an Optimized Pipeline

The script **automl.py** demonstrates how to use TPOT for predicting **Breast Cancer**. It fetches the **UCI Breast Cancer Wisconsin Diagnostic** dataset using the `ucimlrepo` package. If the package is not installed, it can be added using:

```bash
!pip install ucimlrepo
```

### Steps to Build the Pipeline:

1. **Data Preparation**:  
   The dataset is loaded using the `fetch_ucirepo` function, with the target and feature variables separated. The data is converted to **NumPy arrays**, and the target variable is flattened to ensure compatibility. Since the target is categorical, it is **encoded using `LabelEncoder`**.

2. **Data Splitting**:  
   The dataset is split into **training and testing sets** using `train_test_split` from `sklearn`. This ensures that the model's performance can be evaluated on unseen data.

3. **Pipeline Initialization**:  
   The **TPOTClassifier** is initialized with a specified number of generations, population size, and cross-validations. The `n_jobs=-1` parameter ensures that all CPU cores are utilized to speed up the process.

4. **Training and Exporting**:  
   The TPOT model is trained using the `fit()` method. After optimization, the **final code is exported** to a Python file named `BreastCancerAutoML.py`.

---

## Analysis of the Optimized Code

The exported pipeline produced by TPOT includes a **stacking ensemble** of an **XGBoost classifier** followed by a **Bernoulli Naive Bayes** model. The key aspects of the pipeline are:

1. **Stacking with XGBoost**:  
   The first step of the pipeline involves **XGBoost**, a highly efficient gradient boosting algorithm. TPOT fine-tunes parameters such as `learning_rate`, `max_depth`, and `n_estimators` to improve accuracy.

2. **Bernoulli Naive Bayes for Final Prediction**:  
   After the features are transformed by XGBoost, **Bernoulli Naive Bayes** makes the final prediction. This combination enhances performance by leveraging both ensemble learning and probabilistic classification techniques.

3. **Random State Handling**:  
   TPOT ensures reproducibility by setting a **random state** across all steps of the pipeline.

4. **Training and Prediction**:  
   The pipeline is trained on the training data, and predictions are made on the testing data to evaluate performance. TPOT reports the **average cross-validation score** on the training set, which indicates how well the model generalizes.

This optimized pipeline exemplifies how TPOT automates the process of **model selection and hyperparameter tuning**. By stacking models, it achieves a high degree of accuracy with minimal manual effort.

---

TPOT is a powerful tool that simplifies the development of machine learning models. It helps users quickly build high-performance pipelines by exploring multiple models and hyperparameters, making it ideal for both beginners and experts.

