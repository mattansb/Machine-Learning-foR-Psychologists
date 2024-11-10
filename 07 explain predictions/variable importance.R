library(tidymodels)
# library(randomForest)

# We've already seen how to find variable importance metrics (e.g., with the
# {vip} pacakge). But alone, these methods do not usually provide any measure of
# uncertinty in the importance metric.
# Once again, we can use re-sampling for this!


# The data -------------------------------------------------------------------

data("Hitters", package = "ISLR")
?ISLR::Hitters

# Train the model ------------------------------------------------------------

# fit a random forest model
rec <- recipe(Salary ~ .,
              data = Hitters) |>
  step_naomit(Salary) |>
  step_dummy(all_factor_predictors()) |>
  step_normalize(all_numeric_predictors())


rf_spec <- rand_forest(
  mode = "regression", engine = "randomForest",
  # Using the p/3 rule of thumb
  mtry = .cols() / 3
)

rf_wf <- workflow(preprocessor = rec, spec = rf_spec)

rf_fit <- fit(rf_wf, data = Hitters)




# Variable importance --------------------------------------------------------

# Let's use the {vip} pacakge that has several methods to compute variable
# importance - specifically the model-based methods with vip::vi_model(). This
# method gives VIs based on the input model type. See:
?vip::vi_model

rf_eng <- extract_fit_engine(rf_fit)
vip::vi_model(rf_eng)

# We can also plot with vip::vip()
vip::vi_model(rf_eng) |>
  vip::vip(num_features = 1000) + # show all
  theme_bw()


## Uncertenty ---------------------------------

# For a linear regression we might have an analytical or approximate method to
# compute standard-errors. But not here.
# The solution: use resampling!

# split the data
set.seed(111)
cv_folds <- vfold_cv(Hitters, v = 10)

# A little helper function to extract the VIs from the fit-workflow.
extract_rf_vi <- function(wf) {
  extract_fit_engine(wf) |>
    vip::vi_model()
}

# Use fit_resamples() and request the function be applied on each re-sample.
rf_samps <- fit_resamples(rf_wf,
  resamples = cv_folds,
  control = control_resamples(extract = extract_rf_vi)
)

rf_samps


rf_vis <-
  # We can collect the .extracts with
  collect_extracts(rf_samps) |>
  # And unnest the results
  unnest(.extracts) |>
  # Cleanup any missing values (set them to 0)
  complete(id, Variable, fill = list(Importance = 0))
rf_vis


# Let's summarize the results:
rf_vis_summ <- rf_vis |>
  group_by(Variable) |>
  summarise(
    .mean = mean(Importance),
    .std_err = sd(Importance) / sqrt(n())
  ) |>
  ungroup()



# Plot:
rf_vis_summ |>
  mutate(
    # reorder the variable names
    Variable = forcats::fct_reorder(factor(Variable), .mean)
  ) |>
  ggplot(aes(.mean, Variable)) +
  geom_col(fill = "royalblue") +
  geom_errorbar(aes(xmin = .mean - .std_err, xmax = .mean + .std_err),
                width = 0.1) +
  theme_classic()
# (Look at CHmRum vs RBI or Hits)
