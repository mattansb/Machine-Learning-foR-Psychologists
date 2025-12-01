library(tidymodels)
# library(kknn)

# The data ----------------------------------------------------------------

data("Hitters", package = "ISLR")
ISLR::Hitters
Hitters <- tidyr::drop_na(Hitters, Salary)

# Split:
set.seed(20251201)
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
Hitters.test <- testing(splits)
# Our data is REALLY SMALL such that splitting the data to train and test might
# leave us with very small datasets.

rec <- recipe(Salary ~ ., data = Hitters.train) |>
  step_dummy(all_factor_predictors(), one_hot = TRUE) |>
  # Let's add an interaction here:
  step_interact(~ HmRun:starts_with("League")) |>
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors())
# IMPORTANT! scale the variable pre-fitting

# PCR ---------------------------------------------------------------------
# This is a method that use PCA as a first step for predicting y.
# IMPORTANT- this is an UNSUPERVISED method for dimension reduction - since PCA
# don't uses a response variable when building the components. There are other
# methods, which uses the response (hence, SUPERVISED), for using a similar
# goal- predicting y from components

rec
# Already has a scaling step and centering step
# If we didn't we would have to add
# step_pca(..., options = list(center = TRUE, scale. = TRUE))

# There are two arguments that can be used to control how many PCs to save:
# - num_comp: the number of components
# - threshold: what proportion of variance should be saved?
# * Note that the predictors should be all be re-scaled _prior_ to the PCA step.
pcr_rec <- rec |>
  # We will tune the PCA step!
  step_pca(all_numeric_predictors(), num_comp = tune())


linreg_spec <- linear_reg(mode = "regression", engine = "lm")

linreg_wf <- workflow(preprocessor = pcr_rec, spec = linreg_spec)


## Tune ------------------------------------

# Using 5-fold CV:
set.seed(20251201)
cv_10folds <- vfold_cv(Hitters.train, v = 10)

pcr_grid <- grid_regular(
  num_comp(range = c(1, 7)),

  levels = 7
)

# Tune the model
pcr_tuned <- tune_grid(
  linreg_wf,
  resamples = cv_10folds,
  grid = pcr_grid,
  # Default metrics: rsq, rmse
)

autoplot(pcr_tuned)

(best_pcr <- select_best(pcr_tuned, metric = "rmse"))


## The final model --------------------

pcr_fit <- linreg_wf |>
  finalize_workflow(parameters = best_pcr) |>
  fit(data = Hitters.train)

# These are the coefficients of the PCs
extract_fit_engine(pcr_fit) |> coef()

# With a little help we can obtain the coefficients of the original features:
extract_pcr_coef <- function(x, which = c("coef", "vcov")) {
  stopifnot(
    "x is not a workflow" = inherits(x, "workflow"),
    "the model spec is not linear_reg" = inherits(
      x_spec <- extract_spec_parsnip(x),
      c("linear_reg", "logistic_reg")
    ),
    "last step in recipe is not step_pca" = inherits(
      x_step_pca <- tail(extract_recipe(x)$steps, 1)[[1]],
      "step_pca"
    )
  )

  which <- match.arg(which)

  x_prcomp <- x_step_pca$res
  rotation <- x_prcomp$rotation
  # we only want the first k of these:
  loadings_pcr <- rotation[, 1:x_step_pca$num_comp]

  # get the coefficients and standard errors on the PC scales:
  pcr_eng <- extract_fit_engine(pcr_fit)
  b <- coef(pcr_eng)
  V <- vcov(pcr_eng)

  if (which == "coef") {
    # These are the coefficients on the original scale
    drop(b[-1] %*% t(loadings_pcr))
  } else {
    # These are the standard errors
    loadings_pcr %*% V[-1, -1] %*% t(loadings_pcr)
  }
}

extract_pcr_coef(pcr_fit)
# Note the features were all scaled, so these values are partially standardized
# coefficients (they're not scaled with respect to y, only to X).

# PLS ---------------------------------------------------------------------
# We will now address the same problem using another dimension reduction method-
# PLS. Importantly, PCR is an UNSUPERVISED method and PLS is a SUPERVISED
# method. PLS counts for *Partial Least Squares*.

# install.packages("BiocManager", repos = "https://cloud.r-project.org")
# BiocManager::install("mixOmics")
# library(mixOmics)
library(plsmod)

pls_spec <- pls(
  mode = "regression",
  num_comp = tune()
)

pls_wf <- workflow(preprocessor = rec, spec = pls_spec)

## Tune ---------------------------------------------

pls_grid <- grid_regular(
  num_comp(range = c(1, 7)),

  levels = 7
)

# Tune the model
pls_tuned <- tune_grid(
  pls_wf,
  resamples = cv_10folds,
  grid = pls_grid,
  # Default metrics: rsq, rmse
)


autoplot(pls_tuned)

(best_pls <- select_best(pls_tuned, metric = "rmse"))


## The final model --------------------

pls_fit <- pls_wf |>
  finalize_workflow(parameters = best_pls) |>
  fit(data = Hitters.train)

# Here too we need a little help
extract_pls_coef <- function(x) {
  stopifnot(
    "x is not a workflow" = inherits(x, "workflow"),
    "the model spec is not linear_reg" = inherits(
      x_eng <- extract_fit_engine(x),
      "mixo_spls"
    )
  )

  sd_y <- attr(x_eng$Y, "scaled:scale")

  pr <- predict(x_eng, newdata = x_eng$X[1, , drop = FALSE])

  B.hat <- pr$B.hat
  B.hat[,, dim(B.hat)[3]] * sd_y
}

extract_pls_coef(pls_fit)


## Compare ---------------------------

cbind(PCR = extract_pcr_coef(pcr_fit), PLS = extract_pls_coef(pls_fit))


augment(pcr_fit, new_data = Hitters.test) |> rsq(Salary, .pred)
augment(pls_fit, new_data = Hitters.test) |> rsq(Salary, .pred)
# In this case pcr out performed pls.

# Using PCA in other methods ----------------------------------

# What if I want to use KNN??
# We can still use PCA as part of our recipe.

## Tune a PCR KNN --------------------------

knn_spec <- nearest_neighbor(
  mode = "regression",
  engine = "kknn",
  neighbors = tune()
)

knn_wf <- workflow(preprocessor = pcr_rec, spec = knn_spec)

knn_grid <- expand_grid(
  num_comp = 1:7,
  neighbors = c(1, 2, 5, 10, 20)
)

# Tune the model
knn_tuned <- tune_grid(
  knn_wf,
  resamples = cv_10folds,
  grid = knn_grid,
  # Default metrics: rsq, rmse
)


autoplot(knn_tuned)


(onese_knn <- select_by_one_std_err(
  knn_tuned,
  num_comp,
  desc(neighbors),
  metric = "rmse"
))
# Note this is a different value than
rbind(best_pcr, best_pls)


## The final model --------------------

knn_fit <- fit(
  finalize_workflow(knn_wf, onese_knn),
  data = Hitters.train
)


## Compare -------------------

Hitters.test_predictions <- augment(knn_fit, new_data = Hitters.test)

Hitters.test_predictions |> rsq(Salary, .pred)
# KNN with PCA is better!

# How about KNN without PCA?
