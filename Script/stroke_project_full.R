# stroke_project_full.R
# R version of the full pipeline for the stroke dataset
# Required packages:
# install.packages(c("tidyverse","caret","data.table","pROC","xgboost","randomForest","gbm","ROCR","ranger","patchwork"))

library(tidyverse)
library(caret)
library(data.table)
library(pROC)
library(randomForest)
library(xgboost)
library(ranger)
library(patchwork)

set.seed(42)

# 1) Read data
df <- fread("healthcare-dataset-stroke-data.csv") %>% as_tibble()

# Quick snapshot
cat("Rows:", nrow(df), "Columns:", ncol(df), "\n")
print(names(df))
print(table(df$stroke))

# 2) Basic cleaning & type conversion
df <- df %>%
  mutate(
    gender = as.factor(gender),
    ever_married = as.factor(ever_married),
    work_type = as.factor(work_type),
    Residence_type = as.factor(Residence_type),
    smoking_status = as.factor(smoking_status),
    hypertension = as.factor(hypertension),
    heart_disease = as.factor(heart_disease),
    stroke = as.factor(stroke)
  )

# 3) Missingness overview
missing_table <- sapply(df, function(x) sum(is.na(x)))
print(missing_table)

# 4) Impute BMI median within group (example: median BMI overall or by gender)
# Here we use overall median (or you can use grouping)
bmi_median <- median(df$bmi, na.rm=TRUE)
df$bmi <- ifelse(is.na(df$bmi), bmi_median, df$bmi)

# 5) Feature engineering - (if needed)
# e.g., bucket age into categories (optional)
df <- df %>%
  mutate(age_group = case_when(
    age < 18 ~ "child",
    age >= 18 & age < 35 ~ "young_adult",
    age >= 35 & age < 60 ~ "adult",
    age >= 60 ~ "senior"
  )) %>% mutate(age_group = factor(age_group, levels=c("child","young_adult","adult","senior")))

# 6) Train-test split (stratified by stroke)
train_index <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

# 7) Preprocessing via caret
# We'll one-hot encode categorical vars and center/scale numeric variables
prep_recipe <- preProcess(train %>% select(age, avg_glucose_level, bmi),
                          method = c("center","scale"))

train_num <- predict(prep_recipe, train %>% select(age, avg_glucose_level, bmi))
test_num  <- predict(prep_recipe, test %>% select(age, avg_glucose_level, bmi))

# Create model frames with dummies
dummies <- dummyVars(~ gender + ever_married + work_type + Residence_type + smoking_status + hypertension + heart_disease + age_group,
                     data = train)
train_cat <- predict(dummies, newdata = train) %>% as.data.frame()
test_cat  <- predict(dummies, newdata = test) %>% as.data.frame()

train_model <- bind_cols(train_num, train_cat) %>% mutate(stroke = train$stroke)
test_model  <- bind_cols(test_num, test_cat) %>% mutate(stroke = test$stroke)

# 8) Baseline model: Logistic regression
ctrl <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary, savePredictions="final")

set.seed(42)
# caret requires the positive class as the first level sometimes; ensure factor levels:
train_model$stroke <- relevel(train_model$stroke, ref = "1") # make '1' positive if needed
test_model$stroke  <- relevel(test_model$stroke, ref = "1")

glm_fit <- train(stroke ~ ., data = train_model,
                 method="glm",
                 family="binomial",
                 metric="ROC",
                 trControl = ctrl)

print(glm_fit)
# Predictions on test
glm_prob <- predict(glm_fit, newdata = test_model, type = "prob")[, "1"]
glm_pred <- predict(glm_fit, newdata = test_model)

glm_roc <- roc(as.numeric(as.character(test_model$stroke)), glm_prob)
cat("Logistic AUC:", auc(glm_roc), "\n")
conf_mat_glm <- confusionMatrix(glm_pred, test_model$stroke, positive="1")
print(conf_mat_glm)

# 9) Random Forest baseline
set.seed(42)
rf_fit <- train(stroke ~ ., data = train_model,
                method = "ranger",
                tuneLength = 5,
                metric="ROC",
                trControl = ctrl)

print(rf_fit)
rf_prob <- predict(rf_fit, newdata = test_model, type = "prob")[, "1"]
rf_pred <- predict(rf_fit, newdata = test_model)
rf_roc <- roc(as.numeric(as.character(test_model$stroke)), rf_prob)
cat("RF AUC:", auc(rf_roc), "\n")
print(confusionMatrix(rf_pred, test_model$stroke, positive="1"))

# 10) XGBoost (caret wrapper)
set.seed(42)
xgb_fit <- train(stroke ~ ., data = train_model,
                 method = "xgbTree",
                 tuneLength = 6,
                 metric="ROC",
                 trControl = ctrl)

print(xgb_fit)
xgb_prob <- predict(xgb_fit, newdata = test_model, type = "prob")[, "1"]
xgb_pred <- predict(xgb_fit, newdata = test_model)
xgb_roc <- roc(as.numeric(as.character(test_model$stroke)), xgb_prob)
cat("XGB AUC:", auc(xgb_roc), "\n")
print(confusionMatrix(xgb_pred, test_model$stroke, positive="1"))

# 11) Plot ROC curves
roc_df <- data.frame(
  glm = as.numeric(glm_prob),
  rf = as.numeric(rf_prob),
  xgb = as.numeric(xgb_prob),
  true = as.numeric(as.character(test_model$stroke))
)

# Using pROC to plot
plot(glm_roc, main="ROC Curves - Models")
plot(rf_roc, add=TRUE, col=2)
plot(xgb_roc, add=TRUE, col=3)
legend("bottomright", legend=c(sprintf("Logistic AUC=%.3f", auc(glm_roc)),
                               sprintf("RF AUC=%.3f", auc(rf_roc)),
                               sprintf("XGB AUC=%.3f", auc(xgb_roc))),
       col=1:3, lwd=2)

# 12) Save final model (example: xgboost)
saveRDS(xgb_fit, "xgb_fit_final.rds")

# End of script
