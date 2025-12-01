library(tidymodels)
library(patchwork)

# Note that loo_cv() does not (yet?) play nicely with resampling methods as
# implemented throughout {tidymodels}. You can still use it -- e.g., for OOS
# performance estimation -- but it requires manual code writing.
# Let's do just that.

# The data ----------------------------------------------------------------
# The most famous data set in R.

data("mtcars")
?datasets::mtcars

nrow(mtcars)
# The dataset is too small for splitting...
vfold_cv(mtcars, v = 10) # gives only 3 obs. in the validation sets.
bootstraps(mtcars, times = 200) # also give small validation sets.

# The solution - use the WHOLE set as a validation set.
# 1. Fit N models on a subset of size (N-1).
# 2. Predict the outcome for the Nth observation.
# 3. Collect the predictions - we now have a single validation set of size N.
# This method gives highly variable results, but is often better than nothing in
# small samples.

splits <- loo_cv(mtcars)
splits
# We can see each split has 31 samples for training, and 1 left out.

mset_reg <- metric_set(rsq, rmse)


# Setting up a model ------------------------------------------------------

# These steps, for building a model spec, a recipe (a "workflow") are identical
# as what we've seen.

linreg_spec <- linear_reg(mode = "regression", engine = "lm")

rec <- recipe(mpg ~ ., data = mtcars)

linreg_wf <- workflow(preprocessor = rec, spec = linreg_spec)

# Use resampling ----------------------------------------------------------

fit_resamples(linreg_wf, resamples = splits)
#> Error in `check_rset()`:
#> ! Leave-one-out cross-validation is not currently supported with tune.

# We cannot use tune::fit_resamples().

# Or can we...?
class(splits)[1] <- "vfold_cv" # (Is this legal?)

results <- fit_resamples(
  linreg_wf,
  resamples = splits,
  # We don't be computing out metrics for each fold - only once at the end,
  # when we have all predictions. For that, we need to save the predictions:
  control = control_resamples(save_pred = TRUE)
)


preds <- collect_predictions(results)

# If we want to see the predictions with the original data:
mtcars_preds <- bind_cols(
  # get original data in the same order as in `preds`:
  mtcars[preds$.row, ],
  .pred = preds$.pred
)
# For each row, we have the original data + a `.pred` column for the OOS
# estimation!

# Estimate performance ----------------------------------------------------

mtcars_preds |>
  mset_reg(truth = mpg, estimate = .pred)
# Not bad.
# How does this compare to the rsq and the adjusted rsq of the model trained on
# the full data?

linreg_fit <- fit(linreg_wf, data = mtcars)
extract_fit_engine(linreg_fit) |>
  performance::model_performance(metrics = c("R2", "R2_adj", "RMSE"))
# We can see that even the adjusted rsq gives an over-estimates...

(ggplot(mtcars_preds, aes(.pred, mpg)) +
  geom_abline() +
  geom_point() +
  coord_obs_pred(xlim = c(5, 35)) +
  ggtitle("LOO-CV predictions")) +

  (augment(linreg_fit, new_data = mtcars) |>
    ggplot(aes(.pred, mpg)) +
    geom_abline() +
    geom_point() +
    coord_obs_pred(xlim = c(5, 35)) +
    ggtitle("In-sample predictions")) +

  plot_annotation(
    caption = "LOO-CV looks worse, but in-sample is actually overfitting."
  )
