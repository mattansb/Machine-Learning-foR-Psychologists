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
# We will be using data a little differently here since we will be doing 3
# things along the way:
# 1. Tune hyper parameters
# 2. Compare models (and select between them)
# 3. Test the final model
# We want a "clean" dataset for each type of work that is independent from the
# data used in previous steps.

# So let's split the data in 3 parts:
# 35% for tuning, 35% for comparing, and the remaining (30%) for testing:
p <- c(0.35, 0.35)

n <- nrow(Auto)
idx <- sample.int(n)
k_start <- floor(n * c(0, cumsum(p))) + 1
k_end <- floor(n * c(cumsum(p), 1))

Auto.tune <- Auto[sort(idx[k_start[1]:k_end[1]]), ]
Auto.compare <- Auto[sort(idx[k_start[2]:k_end[2]]), ]
Auto.test <- Auto[sort(idx[k_start[3]:k_end[3]]), ]

# As you can see, these splits require a lot of data.
# In smaller samples we might use loo-cv for comparing models, or collapse
# decision nodes.

## Get resampled results --------------------------------------------------

### Model 1: Linear regression --------------------------------------------

rec1 <- recipe(mpg ~ horsepower + weight, data = Auto.compare)

linreg_spec <- linear_reg(mode = "regression", engine = "lm")

linreg1_wf <- workflow(preprocessor = rec1, spec = linreg_spec)

# (No tuning needed.)

# Split comparison data (why?):
cv_compare <- vfold_cv(Auto.compare, v = 10)
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
  metrics = mset_reg
)


# get a summary of the metrics across resamples:
collect_metrics(linreg1_oos)
# We can also get the raw data:c data (per-resample):
collect_metrics(linreg1_oos, summarize = FALSE)
collect_metrics(linreg1_oos, summarize = FALSE, type = "wide")


### Model 2: Linear regression --------------------------------------------

rec2 <- recipe(mpg ~ ., data = Auto.compare) |>
  step_rm(name) |>
  step_novel(cylinders, origin) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_numeric_predictors())

# (No tuning needed.)

linreg2_wf <- workflow(preprocessor = rec2, spec = linreg_spec)


linreg2_oos <- fit_resamples(
  linreg2_wf,
  resamples = cv_compare, # We are using the SAME resamples!
  metrics = mset_reg
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

# Split tuning data for tuning(!):
cv_tune <- vfold_cv(Auto.tune, v = 5)

knn_grid <- grid_regular(
  neighbors(range = c(10, 100)),
  levels = 7
)

knn_tuner <- tune_grid(
  wf_knn,
  # Note we're using the train-split data
  resamples = cv_tune,
  grid = knn_grid,
  metrics = mset_reg
)


autoplot(knn_tuner)

(k_1SE <- select_by_one_std_err(knn_tuner, desc(neighbors), metric = "mae"))


#### Fit the final model -----------------------

wf_knn <- finalize_workflow(wf_knn, k_1SE)
wf_knn

knn_oos <- fit_resamples(
  wf_knn,
  # Note we're using the same splits!
  resamples = cv_compare,
  metrics = mset_reg
)


## Compare the resampled performance ------------------------------------------------
# We're actually doing two things with this:
# 1. We are getting CV estimates of the OOS performance of the models.
# 2. Because we used the *same* folds, we can compare the models in a paired
#    fashion!

### Plot -----------------

cv_results <- bind_rows(
  linear1 = collect_metrics(linreg1_oos, summarize = FALSE),
  linear2 = collect_metrics(linreg2_oos, summarize = FALSE),
  KNN = collect_metrics(knn_oos, summarize = FALSE),

  .id = "model"
) |>
  mutate(
    model = factor(model, levels = c("linear1", "linear2", "KNN")),
  )

cv_summary <- bind_rows(
  linear1 = collect_metrics(linreg1_oos),
  linear2 = collect_metrics(linreg2_oos),
  KNN = collect_metrics(knn_oos),
  .id = "model"
)


cv_results |>
  ggplot(aes(model, .estimate)) +
  facet_wrap(vars(.metric), scales = "free_y") +
  # fold-data
  geom_line(aes(group = id)) +
  # summary
  geom_pointrange(
    aes(y = mean, ymin = mean - std_err, ymax = mean + std_err),
    data = cv_summary,
    color = "red"
  )


### Contrast ----------------

cv_compareare_lin2.knn <- cv_results |>
  pivot_wider(names_from = model, values_from = .estimate) |>
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


# Investigating prediction errors ----------------------------------------------------------
# It can often be interesting to see not only which model is better, but also
# where different models fail.

# Let's train the selected model(s?) on tune+compare datasets
Auto.train <- Auto[sort(idx[k_start[1]:k_end[2]]), ]
nrow(Auto.train)

# Note we're using the full train data
linreg2_fit <- fit(linreg1_wf, data = Auto.train)
knn_fit <- fit(wf_knn, data = Auto.train)

# Note we're using the test data
linreg2_predictions <- augment(linreg2_fit, new_data = Auto.test)
knn_predictions <- augment(knn_fit, new_data = Auto.test)

## Sub-sample performance -------------------------------------

# We can also look are group performance indices:
linreg2_predictions |>
  group_by(origin, mpg >= median(mpg)) |>
  mae(mpg, .pred)

ggplot(
  linreg2_predictions,
  aes(origin, mpg - .pred, fill = mpg >= median(mpg))
) +
  geom_hline(yintercept = 0) +
  geom_violin()
# We can seen that the model tends to fail the most for cars from Europe with
# high MPG, and least for American cars with low MPG.

# etc...
linreg2_predictions |>
  group_by(cut_number(horsepower, 3)) |>
  mae(mpg, .pred)

## Comparing error distributions --------------------------------

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

# For classification problems, we can also compare errors with
?mcnemar.test()
