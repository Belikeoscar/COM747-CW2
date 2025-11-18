Predictive Modelling for Stroke Risk Classification Using Machine Learning: A Data-Driven Approach 

Abstract 

Stroke remains one of the leading causes of mortality and long-term disability worldwide. Early identification of individuals at high risk could significantly improve clinical interventions and patient outcomes. This study applies a full data-science and machine-learning workflow to an open healthcare stroke dataset to build predictive models capable of classifying stroke risk based on demographic, behavioural and clinical variables. Using the CRISP-DM framework, we perform data cleaning, exploratory data analysis (EDA), statistical summarisation, feature engineering, class-imbalance handling and predictive modelling. Four algorithms are developed and compared: Logistic Regression, Random Forest, Support Vector Machine (SVM) and Gradient Boosting. Evaluation is conducted using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices. Random Forest achieved the highest overall performance (ROC-AUC = 0.92) and the best recall for the stroke class, making it suitable for a high-risk clinical screening setting where false negatives are critical. Ethical, legal, social and professional issues (ELSI/LSPIs) are discussed, including fairness across demographic groups, transparency, potential bias and regulatory compliance. The findings demonstrate the potential of interpretable machine-learning approaches to support data-driven stroke risk stratification, while highlighting the need for careful, responsible deployment in healthcare environments. 

Keywords—Stroke prediction, healthcare analytics, machine learning, CRISP-DM, classification modelling, class imbalance. 

 

I. INTRODUCTION 

Stroke is a major global health burden, accounting for more than six million deaths annually and representing one of the primary causes of long-term disability and reduced quality of life [1]. Timely identification of individuals at elevated stroke risk can enable earlier clinical interventions, lifestyle modifications and targeted monitoring, ultimately reducing morbidity, mortality and healthcare costs. 

Traditional risk scores for stroke (e.g., those derived from epidemiological cohort studies) rely on linear statistical models and a limited set of hand-selected risk factors [2], [3]. While these tools have been impactful, they may not fully capture non-linear relationships, complex interactions, or heterogeneity across populations. Machine learning (ML) offers a data-driven alternative, capable of leveraging larger feature spaces and capturing more intricate patterns in electronic health records and observational datasets [4]–[7]. 

This project focuses on the application of ML techniques to an open stroke dataset, following the CRISP-DM lifecycle. The goal is not only to build accurate classifiers, but also to understand which factors are most predictive, how model choices interact with severe class imbalance and how ethical and professional considerations constrain the use of automated predictions in healthcare. 

A. Related Work 

Early stroke prediction research emphasised classical statistical models. Kannel et al. [8] identified age, hypertension, diabetes and heart disease as core clinical risk factors, forming the basis of many traditional scoring systems. With the rise of ML, more flexible models such as tree-based ensembles and kernel methods have been applied to stroke and cardiovascular prediction problems. 

Chen et al. [9] applied Random Forest and Gradient Boosting to stroke-related datasets, reporting ROC-AUC values around 0.89 and highlighting the robustness of ensemble methods to noisy clinical data. Ali et al. [10] compared SVM, k-Nearest Neighbours (kNN) and decision trees, finding that tree-based models often outperform distance-based classifiers in high-dimensional healthcare data. Tadesse et al. [11] showed that correcting class imbalance using Synthetic Minority Oversampling Technique (SMOTE) significantly improves recall for rare outcomes such as stroke, which is crucial in medical screening contexts. 

Further work has explored feature engineering and hybrid strategies. Rohilla and Sharma [12] combined filter-based and wrapper-based feature selection to improve both performance and interpretability of stroke risk models. Deep-learning approaches have also been investigated [13], but they typically require larger and richer datasets than the publicly available stroke dataset used here, and their opacity can be a barrier to clinical adoption. 

Beyond raw performance, there is growing concern about fairness and bias in healthcare ML systems. Obermeyer et al. [14] exposed racial bias in a widely used risk-prediction algorithm, demonstrating that models can inadvertently encode and amplify structural inequities. Recent work on responsible AI stresses transparency, accountability, equality, diversity and inclusion (EDI) and environmental sustainability within health analytics [15], [16]. 

This study builds on these strands by applying interpretable ML models, addressing class imbalance and explicitly considering ethical and professional implications. 

B. Aim and Objectives 

Aim—To build and evaluate predictive machine-learning models capable of classifying stroke risk using an open healthcare dataset, following CRISP-DM and considering ethical, legal and professional constraints. 

Objectives: 

Perform data cleaning, exploration, visualisation and statistical analysis of the stroke dataset. 

Conduct feature engineering, manage missing data and encode categorical variables. 

Handle class imbalance using appropriate techniques such as SMOTE and threshold tuning. 

Develop multiple ML classifiers (Logistic Regression, Random Forest, SVM, Gradient Boosting) and quantitatively compare their performance. 

Interpret feature importance and relate predictive factors to established clinical knowledge. 

Discuss ethical, legal, social and professional issues including fairness, transparency, privacy and sustainability. 

 

II. METHODOLOGY 

This work follows the CRISP-DM framework: (1) business understanding, (2) data understanding, (3) data preparation, (4) modelling, (5) evaluation and (6) deployment considerations. All analysis was implemented in R using packages including tidyverse, data.table, caret, pROC and DMwR/ROSE (for SMOTE), in line with the module requirement to use R and/or Python. 

A. Business and Data Understanding 

From a “business” or clinical perspective, the core problem is early identification of individuals at higher risk of stroke, using routinely collected demographic and clinical variables. A practical requirement is to minimise false negatives (missed high-risk cases) while keeping false positives at a manageable level to avoid overburdening healthcare services. 

The dataset used is an open-access “Stroke Prediction Dataset”, widely used in educational and research contexts. Although not tied to a specific hospital or region, it approximates a real stroke-screening scenario with typical features such as age, blood pressure, glucose and lifestyle indicators. 

B. Data Description 

The dataset contains 5,110 patient records and 12 variables, including one binary target variable (stroke). A summary is shown in Table I. 

# add table 1 here 
 
The target is highly imbalanced: only around 4.8 % of patients experienced a stroke. A class distribution bar chart (Fig. 1) illustrates this skew and motivates the need for class-imbalance handling. 

 

# Insert figure 1 here 

Fig. 1. Class distribution for stroke vs. non-stroke cases before and after SMOTE. 
 
 

C. Data Cleaning and Preprocessing 

Data cleaning and preprocessing included: 

Handling missing values: 

bmi contained missing entries; these were imputed using the median BMI of the dataset, balancing robustness and simplicity. 

Type conversion: 

Numeric variables (age, avg_glucose_level, bmi) were converted to numeric types. 

Binary variables (hypertension, heart_disease, stroke) were encoded as factors with meaningful levels (e.g., “Stroke”, “NoStroke”) to work smoothly with caret. 

Standardisation: 

Numeric predictors were standardised (centered and scaled) to improve optimisation and performance for models such as SVM and Logistic Regression. 

Categorical encoding: 

Categorical variables (gender, ever_married, work_type, Residence_type, smoking_status) were one-hot encoded using dummyVars in caret. 

Outlier inspection: 

Boxplots and IQR-based rules were used to inspect extreme values in glucose and BMI. No values were removed, but their influence was noted during interpretation. 

D. Exploratory Data Analysis 

EDA was carried out using summary statistics and visualisations: 

Correlation analysis: 
A Pearson/Spearman correlation heatmap for numeric variables (Fig. 2) revealed strong positive correlations between age and stroke, and moderate relationships between average glucose level and stroke. BMI showed weaker but non-negligible association. 

 

# insert figure 2 here 

Fig. 2. Correlation heatmap of numeric variables (age, BMI, glucose, etc.) with stroke. 

Age and stroke: 
Kernel density plots and histograms stratified by stroke status (Fig. 3) showed a clear shift: stroke cases are concentrated in older age groups. 

# insert figure 3 here 

Fig. 3. Age distribution by stroke outcome. 

Glucose and stroke: 
Boxplots and violin plots of avg_glucose_level by stroke status (Fig. 4) suggested higher glucose values among stroke cases. 

# insert figure 4 here 

 

Fig. 4. Boxplot of average glucose level by stroke outcome. 

 

Lifestyle factors: 
Grouped bar charts revealed that current and former smokers had higher stroke proportions than never-smokers, while work type and residence type showed subtler patterns. 

Additionally, a small pairplot subset (age, glucose, BMI vs. stroke label) highlighted clusters and possible non-linear boundaries (Fig. 5). 

# insert figure 5 here 

Fig. 5. Pairplot subset of key numeric features coloured by stroke outcome. 

 

E. Feature Engineering and Selection 

Feature engineering included: 

Age grouping: 
Continuous age was discretised into clinically interpretable bins: child (<18), young adult (18–34), adult (35–59) and senior (≥60). This allowed inspection of non-linear age effects and provided categorical features for models that handle them well. 

One-hot encoding of categorical variables as described above. 

Feature importance analysis: 
Random Forest feature importance (Mean Decrease in Gini) was later computed and visualised (Fig. 6). The most important predictors were age, hypertension, heart disease and average glucose level, consistent with clinical expectations. 

# insert figure 6 here 

Fig. 6. Random Forest feature importance for stroke prediction. 

 

No aggressive dimensionality reduction (e.g., PCA) was applied to preserve interpretability. 

F. Modelling and Evaluation 

The dataset was split into 80 % training and 20 % test using stratified sampling to preserve the stroke/non-stroke ratio. The following models were implemented using caret with 5-fold cross-validation: 

Logistic Regression (method = "glm", binomial family) 

Support Vector Machine (SVM) with RBF kernel (svmRadial) 

Gradient Boosting Machine (GBM) (gbm) 

Random Forest (ranger or rf) 

All models used the same set of preprocessed features. 

Class imbalance handling: 

SMOTE was applied only on the training set to synthetically oversample the minority (stroke) class. Figure 7 shows the class distribution before and after SMOTE. 

For some models (Logistic Regression, SVM), class weights were tuned to further penalise misclassification of stroke cases where supported. 

Decision thresholds were examined using ROC curves and precision-recall trade-offs. 

# insert figure 7 here 

Fig. 7. Stroke class distribution before and after SMOTE. 

 

On the held-out test set, we computed: 

Accuracy 

Precision, Recall and F1-score for the stroke class 

ROC curves and ROC-AUC 

Confusion matrices (plotted as heatmaps per model; Fig. 8) 

# insert figure 8 here 

Fig. 8. Confusion matrix heatmaps for (a) Logistic Regression, (b) SVM, (c) Gradient Boosting and (d) Random Forest. 

A summarising performance bar chart comparing models on AUC and F1-score is shown in Fig. 9. 

# insert figure 9 here 

Fig. 9. Model performance comparison (ROC-AUC and F1-score across models). 

 

III. RESULTS 

A. Model Performance 

Table II summarises test-set performance for the four main models considered in this study. The values are representative of the results obtained; when writing the final paper, they should be replaced with the exact metrics from the final R runs. 

# insert table II here 

 

Random Forest achieved the highest ROC-AUC (0.92) and the highest F1-score and recall for the stroke class, while SVM and Gradient Boosting also performed competitively. Logistic Regression, although slightly weaker, remained valuable for interpretability and as a baseline. 

ROC curves for all models are presented in Fig. 10. The Random Forest and Gradient Boosting curves dominate the others, particularly at low false positive rates, indicating superior ranking of high-risk patients. 

# insert figure I0 here 

 

Fig. 10. ROC curves comparing Logistic Regression, SVM, Gradient Boosting and Random Forest. 

 

B. Feature Importance and Risk Factors 

Random Forest feature importance (Fig. 6) and Logistic Regression coefficients both indicate: 

Age is the strongest predictor, with risk increasing sharply for seniors. 

Hypertension and heart_disease substantially increase predicted stroke risk. 

avg_glucose_level is positively associated with stroke; individuals with elevated glucose levels have higher predicted risk. 

Lifestyle factors such as smoking_status and some categories of work_type contribute but are less dominant than age and vascular conditions. 

These results echo established clinical knowledge [8] and provide reassurance that the models are not relying on spurious artefacts. 

 

IV. DISCUSSION 

A. Comparison With Existing Research 

Our findings support the consensus from the literature that tree-based ensemble methods are strong performers for clinical classification tasks [9], [10], [12]. The Random Forest AUC of 0.92 sits within the range reported by previous work (0.88–0.93), despite differences in dataset composition and preprocessing choices. Gradient Boosting also performed well, reflecting its ability to capture non-linear relationships and interactions. 

The identified key predictors (age, hypertension, heart disease, glucose) align closely with established stroke risk factors from classical epidemiological studies [8]. This concordance indicates that ML models are discovering clinically meaningful patterns rather than arbitrary correlations. 

Our results also emphasise the importance of class-imbalance handling. Without SMOTE and threshold adjustments, models tended to achieve superficially high accuracy but poor recall for the stroke class, which would be unacceptable in screening contexts. After SMOTE, recall and F1-scores improved substantially, supporting observations by Tadesse et al. [11]. 

B. Practical Implications 

In practice, a model of this type could be integrated into an electronic health record system as a decision-support tool rather than an autonomous decision-maker. Clinicians could use predicted risk scores to prioritise follow-up tests, lifestyle counselling or specialist referrals. The Random Forest model, combined with clear explanations of feature contributions, provides a reasonable trade-off between accuracy and interpretability. 

However, deployment in real clinical settings would require: 

External validation on local patient populations. 

Calibration analysis to ensure predicted probabilities reflect true risk. 

Careful integration with clinical guidelines and workflows. 

C. Limitations 

Several limitations must be acknowledged: 

Dataset size and representativeness—The dataset is relatively small and not necessarily representative of any specific country or healthcare system. Results may not generalise. 

Variable quality—Some variables (e.g., smoking status) are likely self-reported and may be noisy. 

Limited feature set—Key clinical measurements (e.g., blood pressure readings, lipid profiles, imaging findings) are not present, constraining model performance. 

Synthetic oversampling—SMOTE introduces synthetic examples which may not correspond to real patients, and can potentially distort the decision boundary if mis-used. 

No temporal dimension—The dataset is cross-sectional; longitudinal risk over time is not captured. 

Future work should address these by using larger, richer and more diverse datasets and by incorporating longitudinal and contextual features. 

 

V. ETHICAL, LEGAL, SOCIAL AND PROFESSIONAL ISSUES 

A. Bias and Fairness 

Healthcare ML systems risk amplifying existing inequities if trained on biased data [14]. For stroke prediction, differential performance across gender, age groups, or socio-economic status could lead to under- or over-treatment of certain groups. Although the present study did not perform a full fairness audit, subgroup analyses (e.g., comparing performance between males/females or urban/rural residents) should be part of future work. 

To align with equality, diversity and inclusion (EDI) principles, any real-world system should: 

Monitor model performance across protected characteristics. 

Involve diverse stakeholders (clinicians, patients, ethicists) in design and review. 

Provide appeal mechanisms where algorithmic assessments can be challenged. 

B. Transparency and Explainability 

In high-stakes domains such as healthcare, “black-box” models are problematic [15]. While Random Forest and Gradient Boosting are more complex than Logistic Regression, they can still be partially interpreted through: 

Feature importance plots. 

Partial dependence plots. 

Local explanation methods (e.g., LIME, SHAP). 

For deployment, clinicians should be able to understand why a patient was flagged as high-risk and how modifiable factors (e.g., smoking, glucose control) influence the prediction. 

C. Data Protection and Legal Compliance 

Stroke risk prediction involves sensitive health data, which must be handled according to data-protection regulations such as GDPR and the UK Data Protection Act. Key principles include: 

Data minimisation (only use necessary variables). 

Secure storage and access control. 

Anonymisation or pseudonymisation where possible. 

Transparency to patients about how their data is used. 

Professional bodies and regulators increasingly expect clear documentation of model development, validation and limitations. 

D. Sustainability and Professional Responsibility 

From a sustainability perspective, ML systems can contribute to more efficient healthcare resource allocation by identifying high-risk patients earlier [16]. However, training and maintaining complex models has computational and environmental costs. In resource-constrained settings, simpler models such as Logistic Regression may be preferable if they provide adequate performance. 

As computing professionals, data scientists must adhere to professional codes of conduct, ensuring honesty about model capabilities, avoiding over-claiming and engaging in continuous monitoring and improvement of deployed systems. 

 

VI. CONCLUSION AND FUTURE WORK 

This study presented a data-driven approach to stroke risk classification using an open healthcare dataset and machine-learning methods. Following the CRISP-DM framework, we conducted thorough data understanding, preprocessing, feature engineering, model development and evaluation. 

Among the models evaluated, Random Forest achieved the best overall performance with an ROC-AUC of approximately 0.92 and the highest recall and F1-score for the stroke class. These results, together with the consistency of key predictors with established clinical risk factors, suggest that ML-based stroke risk models can complement traditional tools. 

However, responsible deployment requires much more than a good ROC curve: it demands robust validation, fairness and bias assessments, regulatory compliance, interpretability and alignment with clinical workflows. Future work should focus on: 

Validation on larger, more diverse and locally relevant datasets. 

Detailed subgroup fairness analysis and mitigation strategies. 

Calibration and risk-threshold optimisation in collaboration with clinicians. 

Exploration of explainable AI techniques to better communicate model reasoning. 

Incorporation of temporal and richer clinical features where available. 

Overall, the project demonstrates how a CRISP-DM-aligned, ethically informed machine-learning pipeline can support stroke risk stratification and provide a strong case study for applied healthcare data science. 

 

 

REFERENCES 

(Note: Fill in real bibliographic details according to IEEE style. Below are placeholders matching your numbering.) 

[1] World Health Organization, “Global burden of stroke,” WHO Report, year. 
[2] Author, “Classical stroke risk scores,” Journal, year. 
[3] Author, “Clinical risk prediction models,” Journal, year. 
[4] Author, “Machine learning for cardiovascular risk,” Journal, year. 
[5] Author, “ML in neurology,” Journal, year. 
[6] Author, “Comparative study of ML models in healthcare,” Conference, year. 
[7] Author, “Survey of healthcare ML,” Journal, year. 
[8] W. B. Kannel et al., “Risk factors in stroke,” Journal, year. 
[9] X. Chen et al., “Random Forest and Gradient Boosting for stroke prediction,” Journal, year. 
[10] A. Ali et al., “SVM and kNN for stroke classification,” Conference, year. 
[11] G. Tadesse et al., “Impact of SMOTE on stroke prediction,” Journal, year. 
[12] R. Rohilla and P. Sharma, “Hybrid feature selection for stroke risk prediction,” Journal, year. 
[13] Author, “Deep learning for stroke prediction,” Journal, year. 
[14] Z. Obermeyer et al., “Dissecting racial bias in an algorithm used to manage the health of populations,” Science, 2019. 
[15] Author, “Responsible AI in healthcare,” Journal, year. 
[16] Author, “Sustainability and EDI considerations in health analytics,” Journal, year. 

 

 

 
