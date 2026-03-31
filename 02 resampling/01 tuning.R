library(tidymodels)
# library(kknn)
# library(finetune)

# The data and problem ----------------------------------------------------

# Wage dataset contains information about wage and other characteristics of 3000
# male workers in the Mid-Atlantic region.
data(Wage, package = "ISLR")
?ISLR::Wage


# Data Splitting (70%):
set.seed(20251201)
splits <- initial_split(Wage, prop = 0.7)
Wage.train <- training(splits)
Wage.test <- testing(splits)


# We'll use KNN - but we will use resampling methods to find K!

# Tuning a KNN model -----------------------------------

## 1) Specify the model -------------------------------------------

# Define the model
# Note: we are stting neighbors (k) to tune(). This is a placeholder for the
# tuning grid, and will later be replaced by the actual selected value.
knn_spec <- nearest_neighbor(
  mode = "regression",
  engine = "kknn",
  neighbors = tune()
)


# Define the recipe
rec <- recipe(wage ~ ., data = Wage.train) |>
  # Oops - we wouldn't want this! (why?)
  step_rm(logwage) |>
  # Dummy code categorical predictors
  step_dummy(all_nominal_predictors()) |>
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
cv_folds <- vfold_cv(Wage.train, v = 10)
cv_folds
# In each "set" we have 1890 obs. for training, and 210 obs. for validation.

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

mset_reg <- metric_set(rsq, rmse, mae)
mset_reg


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
help.search("^grid_", package = "dials") # See more options here

### Model tuning ----------------------------------

# Running many models can be time consuming. We can use parallel processing to
# speed this up with the {mirai} package.
# See: https://tune.tidymodels.org/articles/extras/optimizations.html#parallel-processing
mirai::daemons(4)

# Tune the model
knn_tuned <- tune_grid(
  knn_wf, # the model to re-fit
  resamples = cv_folds,
  grid = knn_grid,
  metrics = mset_reg
)


#### View results ---------------------
autoplot(knn_tuned)

# We can extract the OOS results:
collect_metrics(knn_tuned)
collect_metrics(knn_tuned, type = "wide")
collect_metrics(knn_tuned, summarize = FALSE) # for each fold


#### Select hyperparameter values ---------------------
# Select best model
(best_knn <- select_best(knn_tuned, metric = "rmse"))

# Or use the one-SE rule
select_by_one_std_err(knn_tuned, desc(neighbors), metric = "rmse")
select_by_one_std_err(knn_tuned, desc(neighbors), metric = "rsq")

# Finalize workflow
knn_final_wf <- finalize_workflow(knn_wf, best_knn)
knn_final_wf


## 3) Fit the final model -------------------------------------
# Using the full training set

knn_final_fit <- fit(knn_final_wf, data = Wage.train)


## 4) Predict and evaluate -------------------------------------------------
# On the test set.

Wage.test_predictions <- augment(knn_final_fit, new_data = Wage.test)
glimpse(Wage.test_predictions)


Wage.test_predictions |>
  mset_reg(truth = wage, estimate = .pred)
# Overall, not amazing...

# Exercises ----------------------------------------------------------------------

# - Fit a KNN model - this time include all predictors (except...?)!
#   Use the FULL dataset (without splitting to train/test sets.)
# - Tune the model
#   A. define a grid of K values
#   B. use a metric(s) of your choice
#     https://yardstick.tidymodels.org/reference/index.html
#   C. Use the following resampling methods:
#     1. With 50 bootstrap samples
(bootstrap_samps <- bootstraps(Wage, times = 50))
# (Note that the validation set is not always of the same size!)
#     2. 10 repeated 5-fold CV:
(cv_repeated_folds <- vfold_cv(Wage, v = 5, repeats = 10))
#   D. Select K using best / one-SE rule.
#     How did the resampling methods differ in their results?
