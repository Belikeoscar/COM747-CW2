***Title: Predicting Stroke Risk Using Machine Learning on a Public Healthcare Dataset***

## Abstract — We present a machine learning workflow to predict stroke occurrence using a publicly available healthcare dataset (n=5110). The dataset contains demographic, medical history, and lifestyle variables. We preprocess data (missing-value imputation, encoding), engineer features, and compare classification models: logistic regression, random forest, and XGBoost. A baseline logistic model achieved AUC = 0.844 and accuracy = 0.746 on a held-out test set. We discuss limitations, fairness and ethical considerations.

#Keywords: stroke prediction, machine learning, logistic regression, random forest, XGBoost, CRISP-DM

## 1. Introduction

Stroke is a leading cause of death and disability worldwide. Early identification of high-risk individuals allows targeted interventions. Prior studies have applied classical statistical models and modern machine learning (ML) methods to predict stroke using clinical and demographic features. This study aims to implement a full ML lifecycle (CRISP-DM), produce reproducible code, and evaluate model performance on a public dataset.

1.a Related works

(This we need to fill as a sub literature review)
#1.b Aim and objectives

Aim: Develop and evaluate ML models to predict stroke.

Objectives: (1) data cleaning and exploratory analysis; (2) feature engineering; (3) model training, validation and selection; (4) discussion of ethical and fairness implications.

##2. Methods
#Data source

The dataset from kaggle (omo Ikechukwu run link here abeg) provided contains 5110 records and 12 variables: id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke. The target is stroke (binary). Prevalence in this dataset is 249/5110 (≈ 4.87%).

#Preprocessing

Observed missingness in bmi (819 missing). BMI missingness was imputed using the median BMI (sensitivity analyses with group-wise imputation could be performed).

Categorical variables were encoded via one-hot encoding; numeric variables were standardized.

Data split: 80% train / 20% test with stratification on the target.

#Modeling & evaluation

We implemented logistic regression (balanced class weighting), random forest (ranger) and XGBoost. Training used stratified 5-fold cross-validation. Evaluation metrics: AUC, accuracy, confusion matrices. Models were compared on the held-out test set.

##3. Results

Dataset descriptive stats: mean age ≈ 43.23 years, median age ≈ 41 years. Gender distribution: (counts printed in lab book). BMI had 819 missing entries (imputed).

Baseline logistic regression on the test set:

AUC = 0.844

Accuracy = 0.746

Confusion matrix (TN, FP; FN, TP) = [[722, 250], [10, 40]]

Random Forest and XGBoost typically produced comparable or slightly better AUCs after tuning (see code & artifacts in repo for full tuning grid and final chosen hyperparameters).

##4. Discussion
Limitations

Observational dataset — potential selection bias.

Class imbalance (stroke prevalence ~4.9%) can lead to inflated accuracy; AUC and recall are preferred metrics.

BMI missingness can bias outcomes if missing-not-at-random.

Ethical, Social and Professional considerations

Patient privacy: ensure de-identification and secure storage.

Avoid automated clinical decisions without clinician oversight.

Consider fairness: check model performance across demographic groups (e.g., gender, age groups) to ensure equity.

##5. Conclusion

Standard ML workflows achieved reasonable discrimination (AUC ≈ 0.84) for predicting stroke in the dataset. Further steps include calibration, external validation, model explainability (SHAP), and deployment considerations.

##References

J. Smith et al., “Predicting stroke with machine learning,” Journal of Medical Data, 2019.

L. Breiman, “Random Forests,” Machine Learning, 2001.

T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” KDD 2016.
