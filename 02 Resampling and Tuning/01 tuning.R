library(tidymodels)
# library(kknn)
# library(finetune)

# The data and problem ----------------------------------------------------

# Smarket dataset contains daily percentage returns for the S&P 500 stock index
# between 2001 and 2005 (1,250 days).
data(Smarket, package = "ISLR")
?ISLR::Smarket
# For each date, the following vars were recorded:
# - Lag1--Lag5 - percentage returns for each of the five previous trading days.
# - Volume - the number of shares traded on the previous day(in billions).
# - Today - the percentage return on the date in question.
# - Direction - whether the market was Up or Down on this date.

# To make life easier, we will relevel the factor so that the positive class is
# FIRST (which is the default behavior in {yardstick}).
Smarket$Direction <- relevel(Smarket$Direction, ref = "Up")

# Assume the following classification task on the Smarket data:
# predict Direction (Up/Down) using the features Lag1 and Lag2.
# If we are not sure how Direction is coded we can use levels():
levels(Smarket$Direction)

table(Smarket$Direction)
# The base rate probability:
table(Smarket$Direction) |> proportions()


# Data Splitting (70%):
set.seed(1234)
splits <- initial_split(Smarket, prop = 0.7)
Smarket.train <- training(splits)
Smarket.test <- testing(splits)


# We'll use KNN - but we will use resampling methods to find K!

# Tuning a KNN model -----------------------------------

## 1) Specify the model -------------------------------------------

# Define the model
# Note: we are stting neighbors (k) to tune(). This is a placeholder for the
# tuning grid, and will later be replaced by the actual selected value.
knn_spec <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  neighbors = tune()
)


# Define the recipe
rec <- recipe(Direction ~ ., data = Smarket.train) |>
  # Keep only Lag predictors
  step_rm(Volume, Today) |>
  # KNN requires standardization of predictors
  step_normalize(all_numeric_predictors())


# Create a workflow
knn_wf <- workflow(preprocessor = rec, spec = knn_spec)
knn_wf


## 2) Tune the hyperparameters --------------------------------------------

# To tune a hyperparameters using resampling methods, we need to define:
# 1) A resmapling method
# 2) What metrics to use for validation
# 3) How to search for different values of hyperparameters.

### Resampling and Metrics ----------------------------------
# We will use 10-fold CV.

# Define the resampling method (10-fold CV)
cv_folds <- vfold_cv(Smarket.train, v = 10)
cv_folds
# In each "set" we have ~788 obs. for training, and ~87 obs. for validation.

# See more methods:
# https://rsample.tidymodels.org/reference/index.html
# https://www.tidymodels.org/learn/work/nested-resampling/
# Note that loo_cv() and nested_cv() do not (yet?) play nicely with resampling
# methods as implemented throughout {tidymodels} since it is often ambiguous how
# to meaningfully compute metric with just one OOS observation/inner-outer CV.
# You can still use it -- e.g., for OOS performance estimation -- but it
# requires manual code writing (see 04 estimating performance with LOO-CV.R).

# For each fold we will compute the out-of-sample performance using the
# following metrics:

mset_class <- metric_set(sensitivity, specificity, f_meas, roc_auc, kap)
mset_class


### Tuning method -------------------------------

# In this course we will be tuning by grid search - via the
?tune_grid
# function, that requires a grid input - predefined candidate values that will
# be used for model fitting and then validation on the validation set(s).
help.search("^tune_", package = c("tune", "finetune")) # See more options here


# Define the tuning grid
knn_grid <- expand_grid(neighbors = c(5, 10, 50, 200))
# Or
knn_grid <- grid_regular(
  neighbors(range = c(5, 200)),
  levels = 4
)

# We can also generate a random grid
?grid_random
# help.search("^grid_", package = "dials") # See more options here

### Model tuning ----------------------------------

# Tune the model
knn_tuned <- tune_grid(
  knn_wf, # the model to re-fit
  resamples = cv_folds,
  grid = knn_grid,
  metrics = mset_class
)


#### View results ---------------------
autoplot(knn_tuned)

# We can extract the OOS results:
collect_metrics(knn_tuned)
collect_metrics(knn_tuned, type = "wide")
collect_metrics(knn_tuned, summarize = FALSE) # for each fold


#### Select hyperparameter values ---------------------
# Select best model
best_knn <- select_best(knn_tuned, metric = "roc_auc")
best_knn

# Or use the one-SE rule
select_by_one_std_err(knn_tuned, desc(neighbors), metric = "roc_auc")
select_by_one_std_err(knn_tuned, desc(neighbors), metric = "specificity")

# Finalize workflow
knn_final_wf <- finalize_workflow(knn_wf, best_knn)
knn_final_wf


## 3) Fit the final model -------------------------------------
# Using the full training set

knn_final_fit <- fit(knn_final_wf, data = Smarket.train)


## 4) Predict and evaluate -------------------------------------------------
# On the test set.

Smarket.test_predictions <- augment(knn_final_fit, new_data = Smarket.test)
glimpse(Smarket.test_predictions)


Smarket.test_predictions |>
  conf_mat(truth = Direction, estimate = .pred_class)


Smarket.test_predictions |>
  mset_class(truth = Direction, estimate = .pred_class, .pred_Up)
# Overall, not amazing...

# Since this is a probabilistic model, we can also look at the ROC curve and AUC:
Smarket.test_predictions |>
  roc_curve(truth = Direction, .pred_Up) |>
  autoplot()


# Exercises ----------------------------------------------------------------------

# - Fit a KNN model - this time include all predictors!
#   Use the FULL dataset (without splitting to train/test sets.)
# - Tune the model
#   A. define a grid of K values
#   B. use a metric(s) of your choice
#     https://yardstick.tidymodels.org/reference/index.html
#   C. Use the following resampling methods:
#     1. With 50 bootstrap samples
(bootstrap_samps <- bootstraps(Smarket, times = 50))
# (Note that the validation set is not always of the same size!)
#     2. 10 repeated 5-fold CV:
(cv_repeated_folds <- vfold_cv(Smarket, v = 5, repeats = 10))
#   D. Select K using best / one-SE rule.
#     How did the resampling methods differ in their results?
