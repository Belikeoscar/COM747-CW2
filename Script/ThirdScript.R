############################################################
# stroke_project_full.R
# Predictive Modelling for Stroke Risk Classification
# with manual random oversampling (no DMwR / SMOTE)
############################################################

# -----------------------------
# 0. Packages (load)
# -----------------------------
# Install once if needed:
# install.packages(c("tidyverse","caret","data.table","pROC",
#                    "randomForest","ranger","gbm","xgboost",
#                    "GGally"))
install.packages("tidyverse")

library(tidyverse)
library(caret)
library(data.table)
library(pROC)
library(randomForest)
library(ranger)
library(gbm)
library(GGally)

set.seed(42)

# -----------------------------
# 1. Load data
# -----------------------------
setwd("C:/Users/user/Documents/GitHub/Data_Science")

df <- fread("healthcare_dataset_stroke_data.csv") %>% as_tibble()

cat("Rows:", nrow(df), "Columns:", ncol(df), "\n")
print(names(df))
print(table(df$stroke))

# -----------------------------
# 2. Cleaning & type conversion
# -----------------------------
df <- df %>%
  mutate(
    age               = as.numeric(age),
    avg_glucose_level = as.numeric(avg_glucose_level),
    bmi               = as.numeric(bmi),
    gender            = as.factor(gender),
    ever_married      = as.factor(ever_married),
    work_type         = as.factor(work_type),
    Residence_type    = as.factor(Residence_type),
    smoking_status    = as.factor(smoking_status),
    hypertension      = as.factor(as.character(hypertension)),
    heart_disease     = as.factor(as.character(heart_disease)),
    stroke = factor(stroke,
                    levels = c(0, 1),
                    labels = c("NoStroke", "Stroke"))
  )

# Missingness overview
missing_table <- sapply(df, function(x) sum(is.na(x)))
print(missing_table)

# Impute BMI with median
bmi_median <- median(df$bmi, na.rm = TRUE)
df$bmi <- ifelse(is.na(df$bmi), bmi_median, df$bmi)
df$bmi <- as.numeric(df$bmi)

# Feature: age_group
df <- df %>%
  mutate(
    age_group = case_when(
      age < 18              ~ "child",
      age >= 18 & age < 35  ~ "young_adult",
      age >= 35 & age < 60  ~ "adult",
      age >= 60             ~ "senior"
    )
  ) %>%
  mutate(age_group = factor(age_group,
                            levels = c("child","young_adult","adult","senior")))

# -----------------------------
# 3. Train / Test split
# -----------------------------
train_index <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

cat("Train size:", nrow(train), " Test size:", nrow(test), "\n")
cat("Train class balance:\n")
print(table(train$stroke))
cat("Test class balance:\n")
print(table(test$stroke))

# -----------------------------
# 4. Manual random oversampling (no SMOTE)
# -----------------------------
oversample_minority <- function(data, target_col) {
  tbl <- table(data[[target_col]])
  maj_class <- names(tbl)[which.max(tbl)]
  min_class <- names(tbl)[which.min(tbl)]
  
  maj_n <- tbl[[maj_class]]
  min_n <- tbl[[min_class]]
  
  maj_data <- data[data[[target_col]] == maj_class, ]
  min_data <- data[data[[target_col]] == min_class, ]
  
  # sample minority with replacement up to majority size
  extra_minority <- min_data[sample(seq_len(nrow(min_data)),
                                    size = maj_n - min_n,
                                    replace = TRUE), ]
  
  upsampled <- bind_rows(data, extra_minority)
  upsampled
}

set.seed(42)
train_up <- oversample_minority(train, "stroke")

cat("After oversampling (train_up class balance):\n")
print(table(train_up$stroke))

# -----------------------------
# 5. Preprocessing: scale + dummies
# -----------------------------
num_vars <- c("age", "avg_glucose_level", "bmi")

prep_recipe <- preProcess(train_up[, num_vars],
                          method = c("center", "scale"))

train_num <- predict(prep_recipe, train_up[, num_vars])
test_num  <- predict(prep_recipe, test[, num_vars])

dummies <- dummyVars(
  ~ gender + ever_married + work_type + Residence_type +
    smoking_status + hypertension + heart_disease + age_group,
  data = train_up
)

train_cat <- predict(dummies, newdata = train_up) %>% as.data.frame()
test_cat  <- predict(dummies, newdata = test)    %>% as.data.frame()

train_model <- bind_cols(train_num, train_cat) %>%
  mutate(stroke = train_up$stroke)

test_model <- bind_cols(test_num, test_cat) %>%
  mutate(stroke = test$stroke)

# Ensure Stroke is positive class (first level)
train_model$stroke <- relevel(train_model$stroke, ref = "Stroke")
test_model$stroke  <- relevel(test_model$stroke,  ref = "Stroke")

# -----------------------------
# 6. TrainControl
# -----------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# -----------------------------
# 7. Models
# -----------------------------

## 7.1 Logistic Regression
set.seed(42)
glm_fit <- train(
  stroke ~ .,
  data = train_model,
  method = "glm",
  family = "binomial",
  metric = "ROC",
  trControl = ctrl
)
print(glm_fit)

## 7.2 Random Forest (ranger)
set.seed(42)
rf_fit <- train(
  stroke ~ .,
  data = train_model,
  method = "ranger",
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl,
  importance = "impurity"
)

## 7.3 SVM (Radial)
set.seed(42)
svm_fit <- train(
  stroke ~ .,
  data = train_model,
  method = "svmRadial",
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl
)
print(svm_fit)

## 7.4 Gradient Boosting (GBM)
set.seed(42)
gbm_fit <- train(
  stroke ~ .,
  data = train_model,
  method = "gbm",
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl,
  verbose = FALSE
)
print(gbm_fit)

# -----------------------------
# 8. Predictions & ROC/AUC
# -----------------------------
glm_prob <- predict(glm_fit, newdata = test_model, type = "prob")[, "Stroke"]
rf_prob  <- predict(rf_fit,  newdata = test_model, type = "prob")[, "Stroke"]
svm_prob <- predict(svm_fit, newdata = test_model, type = "prob")[, "Stroke"]
gbm_prob <- predict(gbm_fit, newdata = test_model, type = "prob")[, "Stroke"]

glm_pred <- predict(glm_fit, newdata = test_model)
rf_pred  <- predict(rf_fit,  newdata = test_model)
svm_pred <- predict(svm_fit, newdata = test_model)
gbm_pred <- predict(gbm_fit, newdata = test_model)

roc_glm <- roc(response = test_model$stroke,
               predictor = glm_prob,
               levels = c("NoStroke", "Stroke"))
roc_rf  <- roc(response = test_model$stroke,
               predictor = rf_prob,
               levels = c("NoStroke", "Stroke"))
roc_svm <- roc(response = test_model$stroke,
               predictor = svm_prob,
               levels = c("NoStroke", "Stroke"))
roc_gbm <- roc(response = test_model$stroke,
               predictor = gbm_prob,
               levels = c("NoStroke", "Stroke"))

cat("AUC - Logistic:", auc(roc_glm), "\n")
cat("AUC - RF      :", auc(roc_rf),  "\n")
cat("AUC - SVM     :", auc(roc_svm), "\n")
cat("AUC - GBM     :", auc(roc_gbm), "\n")

glm_cm <- confusionMatrix(glm_pred, test_model$stroke, positive = "Stroke")
rf_cm  <- confusionMatrix(rf_pred,  test_model$stroke, positive = "Stroke")
svm_cm <- confusionMatrix(svm_pred, test_model$stroke, positive = "Stroke")
gbm_cm <- confusionMatrix(gbm_pred, test_model$stroke, positive = "Stroke")

print(glm_cm)
print(rf_cm)
print(svm_cm)
print(gbm_cm)

# -----------------------------
# 9. Performance summary table
# -----------------------------
perf_df <- tibble(
  Model    = c("Logistic", "Random Forest", "SVM", "GBM"),
  Accuracy = c(glm_cm$overall["Accuracy"],
               rf_cm$overall["Accuracy"],
               svm_cm$overall["Accuracy"],
               gbm_cm$overall["Accuracy"]),
  Recall   = c(glm_cm$byClass["Sensitivity"],
               rf_cm$byClass["Sensitivity"],
               svm_cm$byClass["Sensitivity"],
               gbm_cm$byClass["Sensitivity"]),
  F1       = c(glm_cm$byClass["F1"],
               rf_cm$byClass["F1"],
               svm_cm$byClass["F1"],
               gbm_cm$byClass["F1"]),
  AUC      = c(
    auc(roc_glm),
    auc(roc_rf),
    auc(roc_svm),
    auc(roc_gbm)
  )
)

print(perf_df)

# -----------------------------
# 10. PLOTS
# -----------------------------

## 10.1 Class distribution (overall)
ggplot(df, aes(x = stroke, fill = stroke)) +
  geom_bar() +
  labs(
    title = "Class Distribution: Stroke vs NoStroke (Full Dataset)",
    x = "Stroke Outcome",
    y = "Count"
  ) +
  theme_minimal()

## 10.2 Class distribution before/after oversampling (train only)
before <- train %>%
  count(stroke) %>%
  mutate(Stage = "Before oversampling")

after <- train_up %>%
  count(stroke) %>%
  mutate(Stage = "After oversampling")

oversamp_df <- bind_rows(before, after)

ggplot(oversamp_df, aes(x = stroke, y = n, fill = Stage)) +
  geom_col(position = "dodge") +
  labs(
    title = "Class Distribution Before and After Oversampling (Train Set)",
    x = "Stroke Outcome",
    y = "Count"
  ) +
  theme_minimal()

## 10.3 Correlation heatmap (numeric variables)
num_df <- df %>%
  select(where(is.numeric))

corr_mat <- cor(num_df, use = "complete.obs", method = "pearson")
corr_df <- as.data.frame(as.table(corr_mat))

ggplot(corr_df, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(limits = c(-1, 1), midpoint = 0) +
  labs(
    title = "Correlation Heatmap of Numeric Features",
    x = "",
    y = "",
    fill = "Correlation"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## 10.4 Age distribution by stroke outcome
ggplot(df, aes(x = age, fill = stroke)) +
  geom_density(alpha = 0.4) +
  labs(
    title = "Age Distribution by Stroke Outcome",
    x = "Age (years)",
    y = "Density"
  ) +
  theme_minimal()

## 10.5 Glucose distribution by stroke outcome
ggplot(df, aes(x = avg_glucose_level, fill = stroke)) +
  geom_density(alpha = 0.4) +
  labs(
    title = "Average Glucose Level Distribution by Stroke Outcome",
    x = "Average Glucose Level (mg/dL)",
    y = "Density"
  ) +
  theme_minimal()

## 10.6 Boxplot: glucose by stroke
ggplot(df, aes(x = stroke, y = avg_glucose_level, fill = stroke)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.4) +
  labs(
    title = "Average Glucose Level by Stroke Outcome",
    x = "Stroke Outcome",
    y = "Average Glucose Level (mg/dL)"
  ) +
  theme_minimal()

## 10.7 Random Forest feature importance
rf_varimp <- varImp(rf_fit)$importance %>%
  rownames_to_column("Feature") %>%
  arrange(desc(Overall)) %>%
  slice(1:15)

ggplot(rf_varimp, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Random Forest Feature Importance (Top 15)",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal()

## 10.8 ROC curves (all models)
plot(roc_glm, col = 1, lwd = 2, main = "ROC Curves - All Models")
plot(roc_rf,  col = 2, lwd = 2, add = TRUE)
plot(roc_svm, col = 3, lwd = 2, add = TRUE)
plot(roc_gbm, col = 4, lwd = 2, add = TRUE)

legend("bottomright",
       legend = c(
         sprintf("Logistic (AUC = %.3f)", auc(roc_glm)),
         sprintf("Random Forest (AUC = %.3f)", auc(roc_rf)),
         sprintf("SVM (AUC = %.3f)",          auc(roc_svm)),
         sprintf("GBM (AUC = %.3f)",          auc(roc_gbm))
       ),
       col = 1:4, lwd = 2, cex = 0.8)

## 10.9 Confusion matrix heatmap helper
plot_cm_heatmap <- function(cm, title) {
  df_cm <- as.data.frame(cm$table)
  colnames(df_cm) <- c("Prediction", "Reference", "Freq")
  
  ggplot(df_cm, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", fontface = "bold") +
    scale_fill_gradient(low = "grey30", high = "steelblue") +
    labs(
      title = title,
      x = "True Label",
      y = "Predicted Label"
    ) +
    theme_minimal()
}

plot_cm_heatmap(glm_cm, "Confusion Matrix - Logistic Regression")
plot_cm_heatmap(rf_cm,  "Confusion Matrix - Random Forest")
plot_cm_heatmap(svm_cm, "Confusion Matrix - SVM")
plot_cm_heatmap(gbm_cm, "Confusion Matrix - GBM")

## 10.10 Pairplot subset (age, bmi, glucose)
pair_df <- df %>%
  select(age, bmi, avg_glucose_level, stroke)

ggpairs(
  pair_df,
  aes(color = stroke, alpha = 0.6),
  upper = list(continuous = "points"),
  lower = list(continuous = "smooth"),
  diag  = list(continuous = "densityDiag")
)

## 10.11 Model performance comparison bar chart
perf_long <- perf_df %>%
  pivot_longer(cols = c(Accuracy, Recall, F1, AUC),
               names_to = "Metric",
               values_to = "Value")

ggplot(perf_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(
    title = "Model Performance Comparison",
    x = "Model",
    y = "Score"
  ) +
  theme_minimal()

############################################################
# End of script
############################################################
