library(tidymodels)
# library(kknn)

mirai::daemons(4) # Enable parallel processing

# The data ----------------------------------------------------------------

data("Hitters", package = "ISLR")
ISLR::Hitters
Hitters <- tidyr::drop_na(Hitters, Salary)

# Split:
set.seed(20260119)
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
# Our data is REALLY SMALL such that splitting the data to train and test might
# leave us with very small datasets.

# PCA --------------------------------------------------------------------

# Before we dive into PCR and PLS, let's do a quick PCA on the data to see
# how it works.
#
# In R we can perform PCA using the prcomp() or princomp() functions:
?prcomp
?princomp

# But we will do it with recipes!

pca_rec <- recipe(~., data = Hitters.train) |>
  # Don't want to include Salary in the PCA
  step_rm(Salary) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_normalize(all_numeric_predictors()) |>
  # There are two arguments that can be used to control how many PCs to save:
  # - num_comp: the number of components
  # - threshold: what proportion of variance should be saved?
  # ! Predictors should all be standardized _prior_ to the PCA step.
  # We already did that above, but we could also add
  # step_pca(..., options = list(center = TRUE, scale. = TRUE))
  step_pca(all_numeric_predictors(), num_comp = 5)

# Prepare and bake
pca_rec <- prep(pca_rec)
bake(pca_rec, new_data = Hitters.train)
# This data can then be used in any model we like, plotting, clustering, further
# processed in any way, etc.

# Our PCA accounts for ~85% of the variance in the first 5 components (of 22!)
vars <- pca_rec$steps[[4]]$res$sdev^2
length(vars)
sum(vars[1:5]) / sum(vars)


# PCR ---------------------------------------------------------------------
# This is a method that uses PCA as a first step for predicting y.

rec <- recipe(Salary ~ ., data = Hitters.train) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  # Let's add an interaction here:
  step_interact(~ HmRun:starts_with("League")) |>
  step_normalize(all_numeric_predictors())
# IMPORTANT! scale the features pre-fitting

pcr_rec <- rec |>
  # We will tune the PCA step!
  step_pca(all_numeric_predictors(), num_comp = tune())


linreg_spec <- linear_reg(mode = "regression", engine = "lm")

linreg_wf <- workflow(preprocessor = pcr_rec, spec = linreg_spec)


## Tune ------------------------------------

# Using 10-fold CV:
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
source("_tpoo_utils.R")
extract_pcr_coef(pcr_fit)
# Note the features were all scaled, so these values are partially standardized
# coefficients (they're not scaled with respect to y, only to X).

# PLS ---------------------------------------------------------------------
# We will now address the same problem using another dimension reduction method-
# PLS (Partial Least Squares). Importantly, PCR is an UNSUPERVISED method and
# PLS is a SUPERVISED method.

# install.packages("BiocManager", repos = "https://cloud.r-project.org")
# BiocManager::install("mixOmics")
# library(mixOmics)
# See also ?parsnip::pls()

# Instead of step_pca() we will use step_pls().
# We need to specify the outcome variable here since this is a supervised
# dimension reduction method (unlike PCR).
pls_rec <- rec |>
  step_pls(all_numeric_predictors(), outcome = "Salary", num_comp = tune())

pls_wf <- workflow(preprocessor = pls_rec, spec = linreg_spec)

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
extract_pls_coef(pls_fit)


## Compare ---------------------------

cbind(
  PCR = extract_pcr_coef(pcr_fit),
  PLS = extract_pls_coef(pls_fit)
)

Hitters.test <- testing(splits)

augment(pcr_fit, new_data = Hitters.test) |> rsq(Salary, .pred)
augment(pls_fit, new_data = Hitters.test) |> rsq(Salary, .pred)
# In this case pls out performed pcr.
# (Does it?)

# Using PCA / PLS in other methods ----------------------------------

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
# KNN with PCA is better than PLS.
# But it is though...?

# (What does this code do?)
oos_metrics <- bind_rows(
  PLS = collect_metrics(pls_tuned, summarize = FALSE) |>
    semi_join(best_pls, by = "num_comp"),
  PCR = collect_metrics(pcr_tuned, summarize = FALSE) |>
    semi_join(best_pcr, by = "num_comp"),
  "PCA \u279c KNN" = collect_metrics(knn_tuned, summarize = FALSE) |>
    semi_join(onese_knn, by = c("num_comp", "neighbors")),

  .id = "model"
) |>
  mutate(
    best = model[ifelse(
      .metric == "rmse",
      which.min(.estimate),
      which.max(.estimate)
    )],
    .by = c(id, .metric)
  )


ggplot(oos_metrics, aes(model, .estimate)) +
  facet_wrap(vars(.metric), scales = "free_y") +
  expand_limits(
    rbind(
      data.frame(y = 0, .metric = "rmse"),
      data.frame(y = c(0, 1), .metric = "rsq")
    )
  ) +
  geom_line(aes(group = id, color = best), alpha = .5) +
  stat_summary(color = "red")
