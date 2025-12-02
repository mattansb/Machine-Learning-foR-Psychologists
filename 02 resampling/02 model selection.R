library(tidymodels)
# library(kknn)

# Read mode about model comparisons using resampling here:
# https://www.tmwr.org/compare
# The basic idea behind model comparison with k-folds, is that we can measure
# the OOS performance of each model on each of the folds. Some folds might (by
# chance) be harder / easier to predict for across models, so these are paired
# measures of fit, but each fold is independent (insofar as they _are_ part of a
# set) so we can use rather standard procedures for comparing models across
# paired samples.

# The data -----------------------------------------------------------

data(Auto, package = "ISLR")
?ISLR::Auto
Auto$cylinders <- factor(Auto$cylinders)
Auto$origin <- factor(Auto$origin)

## Data splitting ----------------------------------------

# We will be using the training set for bot tuning (within-model comparison) and
# model comparison (between-model comparison). The test set will be used only at
# the end to get a final estimate of selected model performance.
splits <- initial_split(Auto, prop = 0.7)
Auto.train <- training(splits)
Auto.test <- testing(splits)

## Get resampled results --------------------------------------------------

mirai::daemons(4) # use 4 cores for parallel processing

### Model 1: Linear regression --------------------------------------------

rec1 <- recipe(mpg ~ horsepower + weight, data = Auto.train)

linreg_spec <- linear_reg(mode = "regression", engine = "lm")

linreg1_wf <- workflow(preprocessor = rec1, spec = linreg_spec)

# (No tuning needed.)

# Split data:
cv_compare <- vfold_cv(Auto.train, v = 10)
# We will use these folds for ALL the models - then we can compare the models'
# performance on a fold-wise basis!

# These are the metrics we're interested in:
mset_reg <- metric_set(rsq, mae)


# We will use the fit_resamples() function. This function doesn't actually
# return a fitted model - it just computes a set of performance metrics across
# the resamples.
linreg1_oos <- fit_resamples(
  linreg1_wf,
  resamples = cv_compare,
  metrics = mset_reg,

  # We might want these for later analysis/plots
  control = control_resamples(save_pred = TRUE)
)


# get a summary of the metrics across resamples:
collect_metrics(linreg1_oos)
# We can also get the raw data (per-resample):
collect_metrics(linreg1_oos, summarize = FALSE)
collect_metrics(linreg1_oos, summarize = FALSE, type = "wide")


### Model 2: Linear regression --------------------------------------------

rec2 <- recipe(mpg ~ ., data = Auto.train) |>
  step_rm(name) |>
  step_novel(cylinders, origin) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_numeric_predictors())

# (No tuning needed.)

linreg2_wf <- workflow(preprocessor = rec2, spec = linreg_spec)


linreg2_oos <- fit_resamples(
  linreg2_wf,
  resamples = cv_compare, # We are using the SAME resamples!
  metrics = mset_reg,

  # We might want these for later analysis/plots
  control = control_resamples(save_pred = TRUE)
)


### Model 3: KNN --------------------------------------------

rec3 <- rec2 |>
  step_normalize(all_predictors())

knn_spec <- nearest_neighbor(
  mode = "regression",
  engine = "kknn",
  neighbors = tune()
)

wf_knn <- workflow(preprocessor = rec3, spec = knn_spec)


#### Tune the model --------------------------

knn_grid <- grid_regular(
  neighbors(range = c(5, 100)),
  levels = 7
)

# Note we're using the same data - but a different set of splits! This is a
# single repeated k-fold CV which can protect against overfitting during model
# selection later.
cv_tune <- vfold_cv(Auto.train, v = 10)

knn_tuner <- tune_grid(
  wf_knn,
  resamples = cv_tune,
  grid = knn_grid,
  metrics = mset_reg,

  # We might want these for later analysis/plots
  control = control_grid(save_pred = TRUE)
)


autoplot(knn_tuner)

# Selecting by the one-SE rule also protects us from overfitting:
(k_1SE <- select_by_one_std_err(knn_tuner, desc(neighbors), metric = "mae"))


#### Get OOS results -----------------------

wf_knn <- finalize_workflow(wf_knn, k_1SE)
wf_knn

knn_oos <- fit_resamples(
  wf_knn,
  # Note we're using the same splits!
  resamples = cv_compare,
  metrics = mset_reg
)

# # If we used the same folds for tuning and selection, we can also get the resampled results
# # from the tuning results:
#
# collect_metrics(knn_tuner, summarize = FALSE) |>
#   semi_join(k_1SE, by = ".config")

## Compare the resampled performance ------------------------------------------------
# We're actually doing two things with this:
# 1. We are getting CV estimates of the OOS performance of the models.
# 2. Because we used the *same* folds, we can compare the models in a paired
#    fashion!

### Plot -----------------

cv_results <- bind_rows(
  linear1 = collect_metrics(linreg1_oos, summarize = FALSE),
  linear2 = collect_metrics(linreg2_oos, summarize = FALSE),
  KNN = collect_metrics(knn_tuner, summarize = FALSE) |>
    semi_join(k_1SE, by = ".config"),

  .id = "model"
) |>
  mutate(
    model = factor(model, levels = c("linear1", "linear2", "KNN")),
  )

cv_summary <- bind_rows(
  linear1 = collect_metrics(linreg1_oos),
  linear2 = collect_metrics(linreg2_oos),
  KNN = collect_metrics(knn_tuner) |>
    semi_join(k_1SE, by = ".config"),
  .id = "model"
)


cv_results |>
  group_by(id, .metric) |>
  mutate(
    best = if_else(
      .metric %in% c("rsq"),
      model[which.max(.estimate)],
      model[which.min(.estimate)]
    )
  ) |>
  ggplot(aes(model, .estimate)) +
  facet_wrap(vars(.metric), scales = "free_y") +
  # fold-data
  geom_line(aes(group = id, color = best)) +
  # summary
  geom_pointrange(
    aes(y = mean, ymin = mean - std_err, ymax = mean + std_err),
    data = cv_summary
  )


### Contrast ----------------

cv_compareare_lin2.knn <- cv_results |>
  pivot_wider(
    names_from = model,
    values_from = .estimate,
    id_cols = c(id, .metric)
  ) |>
  group_by(.metric) |>
  mutate(
    # Because these are paired (fold-wise) we can do this:
    diff = linear2 - KNN
  ) |>
  summarise(
    mean_diff = mean(diff),
    se_diff = sd(diff) / sqrt(n()),

    lb = mean_diff - 1.96 * se_diff,
    ub = mean_diff + 1.96 * se_diff
  )

cv_compareare_lin2.knn |>
  # Format results:
  mutate(across(everything(), format)) |>
  glue::glue_data(
    "A diff of {mean_diff} in {.metric}, 95% CI[{lb}, {ub}]"
  )

# Other uses for the resampled results --------------------------------

# It can often be interesting to see not only which model is better, but also
# where different models fail.

# Here we're still looking at the OOS predictions from the training set, but
# we can look at the distribution of errors for the test set as well.
linreg2_predictions <- collect_predictions(linreg2_oos)

knn_predictions <- collect_predictions(knn_tuner) |>
  semi_join(k_1SE, by = ".config")

## 1. Comparing error distributions --------------------------------

ggplot(mapping = aes(mpg - .pred)) +
  geom_vline(xintercept = 0) +
  geom_density(
    aes(fill = "Linear Reg\n(simple)"),
    color = NA,
    alpha = 0.6,
    data = linreg2_predictions
  ) +
  geom_density(
    aes(fill = "KNN"),
    color = NA,
    alpha = 0.6,
    data = knn_predictions
  ) +
  labs(x = expression(mpg - hat(mpg)))
# It seems that the simple linear model gives more negative errors (tends to
# overestimate mpg).

# Have we selected a model?
# Time to see how it performs on the test set!
# ...

## 2. Sub-sample performance -------------------------------------
# Slicing / fairness analyses

# We can also look are group performance indices:

# Add the original variable back in:
Auto.train |>
  slice(linreg2_predictions$.row) |>
  select(-mpg) |>
  bind_cols(linreg2_predictions) |>
  group_by(origin, mpg >= median(mpg)) |>
  rsq(mpg, .pred) |>
  arrange(.estimate)

Auto.train |>
  slice(linreg2_predictions$.row) |>
  select(-mpg) |>
  bind_cols(linreg2_predictions) |>
  ggplot(
    aes(origin, mpg - .pred, fill = mpg >= median(mpg))
  ) +
  geom_hline(yintercept = 0) +
  geom_violin()
# We can seen that the model tends to fail the most for cars from Europe with
# high MPG, and least for American cars with low MPG.

# etc...
Auto.train |>
  slice(linreg2_predictions$.row) |>
  select(-mpg) |>
  bind_cols(linreg2_predictions) |>
  group_by(cut_number(horsepower, 3)) |>
  rsq(mpg, .pred)
