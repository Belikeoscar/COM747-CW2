# ============================================================
# STROKE PREDICTION USING TIDYMODELS (Full Script)
# ============================================================

# Install required packages if needed
# install.packages(c("tidymodels", "themis", "vip"))

library(tidymodels)
library(themis)     # for SMOTE
library(vip)        # variable importance
library(dplyr)
library(ggplot2)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

stroke <- read.csv("healthcare-dataset-stroke-data.csv")

# Ensure stroke is a factor
stroke$stroke <- factor(stroke$stroke, levels=c(0,1))

glimpse(stroke)

# ------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------------------------

set.seed(123)
data_split <- initial_split(stroke, prop = 0.8, strata = stroke)
train_data <- training(data_split)
test_data  <- testing(data_split)

# ------------------------------------------------------------
# 3. CREATE RECIPE (CLEANING + PREPROCESSING)
# ------------------------------------------------------------

stroke_recipe <- recipe(stroke ~ ., data = train_data) %>%
  # Clean missing values
  step_impute_median(all_numeric(), -all_outcomes()) %>%
  step_impute_mode(all_nominal(), -all_outcomes()) %>%

  # Convert categorical to dummy variables
  step_dummy(all_nominal(), -all_outcomes()) %>%

  # Normalize numeric data
  step_normalize(all_numeric(), -all_outcomes()) %>%

  # Balance the dataset
  step_smote(stroke)

stroke_recipe

# Prep recipe for preview
prep(stroke_recipe)

# ------------------------------------------------------------
# 4. RESAMPLING (10-Fold Cross Validation)
# ------------------------------------------------------------

set.seed(123)
cv_folds <- vfold_cv(train_data, v=10, strata = stroke)

# ------------------------------------------------------------
# 5. MODEL SPECIFICATIONS
# ------------------------------------------------------------

# Logistic Regression
log_reg_spec <- logistic_reg(mode = "classification") %>%
  set_engine("glm")

# Random Forest
rf_spec <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# XGBoost
xgb_spec <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Support Vector Machine
svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

# ------------------------------------------------------------
# 6. WORKFLOW SETUP
# ------------------------------------------------------------

log_wf <- workflow() %>%
  add_model(log_reg_spec) %>%
  add_recipe(stroke_recipe)

rf_wf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(stroke_recipe)

xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(stroke_recipe)

svm_wf <- workflow() %>%
  add_model(svm_spec) %>%
  add_recipe(stroke_recipe)

# ------------------------------------------------------------
# 7. HYPERPARAMETER TUNING (EXAMPLE: RANDOM FOREST)
# ------------------------------------------------------------

set.seed(123)
rf_tune <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid = 15,
  metrics = metric_set(roc_auc, accuracy, f_meas, sens, spec)
)

rf_tune

# Plot performance
autoplot(rf_tune)

# ------------------------------------------------------------
# 8. SELECT BEST RF MODEL
# ------------------------------------------------------------

best_rf <- select_best(rf_tune, "roc_auc")
best_rf

final_rf_wf <- finalize_workflow(rf_wf, best_rf)

# ------------------------------------------------------------
# 9. FIT FINAL MODEL ON FULL TRAINING DATA
# ------------------------------------------------------------

final_rf_fit <- final_rf_wf %>%
  fit(data = train_data)

final_rf_fit

# ------------------------------------------------------------
# 10. MODEL EVALUATION ON TEST SET
# ------------------------------------------------------------

rf_predictions <- final_rf_fit %>% 
  predict(test_data) %>% 
  bind_cols(predict(final_rf_fit, test_data, type = "prob")) %>% 
  bind_cols(test_data %>% select(stroke))

# Confusion Matrix
rf_predictions %>% 
  conf_mat(truth = stroke, estimate = .pred_class)

# ROC AUC
roc_auc(rf_predictions, truth = stroke, .pred_1)

# Classification Metrics
rf_predictions %>%
  metrics(truth = stroke, estimate = .pred_class)

# ------------------------------------------------------------
# 11. ROC CURVE PLOT
# ------------------------------------------------------------

rf_predictions %>%
  roc_curve(truth = stroke, .pred_1) %>%
  autoplot()

# ------------------------------------------------------------
# 12. VARIABLE IMPORTANCE
# ------------------------------------------------------------

final_rf_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20)

# ------------------------------------------------------------
# END OF SCRIPT
# ------------------------------------------------------------
