
library(tidymodels)
# library(kknn)

# Read mode about model comparisons using resampling here:
# https://www.tmwr.org/compare

# The data -----------------------------------------------------------

data(Smarket, package = "ISLR")
?ISLR::Smarket

Smarket$Direction <- relevel(Smarket$Direction, ref = "Up")

# Setting up CV folds ------------------------------------------
# We will use these folds for all the models - then we can compare the models'
# preformance on a fold-wise basis!

folds_10 <- vfold_cv(Smarket, v = 10)

# These are the metrics we're interested in:
mset_class <- metric_set(j_index, sensitivity, roc_auc)

# Fit some models --------------------------------------------------

## Train 1: Logistic regression --------------------------------------------

rec1 <- recipe(Direction ~ ., 
               data = Smarket)

log_spec <- logistic_reg(mode = "classification", engine = "glm")

wf_log1 <- workflow(preprocessor = rec1, spec = log_spec)


# We will use the fit_resamples() function. This function doesn't actually
# return a fitted model - it just computes a set of performance metrics across
# the resamples.
oos_logistic1 <- fit_resamples(
  wf_log1,
  resamples = folds_10,
  metrics = mset_class
)

collect_metrics(oos_logistic1)
collect_metrics(oos_logistic1, type = "wide", summarize = FALSE)





## Train 2: Logistic regression --------------------------------------------

rec2 <- recipe(Direction ~ Lag1 + Lag2 + Lag3, 
               data = Smarket)

wf_log2 <- workflow(preprocessor = rec2, spec = log_spec)

oos_logistic2 <- fit_resamples(
  wf_log2,
  resamples = folds_10, # We are using the SAME resamples!
  metrics = mset_class
)


## Train 3: KNN --------------------------------------------

rec3 <- rec1 |> 
  step_normalize(all_predictors())

knn_spec <- nearest_neighbor(
  mode = "classification", engine = "kknn",
  neighbors = tune()
)

wf_knn <- workflow(preprocessor = rec3, spec = knn_spec)


### Tune the model --------------------------

knn_grid <- grid_regular(
  neighbors(range = c(20, 300)), 
  levels = 5
)

knn_tuner <- tune_grid(
  wf_knn,
  resamples = folds_10, # We could use other splits here
  grid = knn_grid, 
  metrics = mset_class
)


autoplot(knn_tuner)

k_1SE <- select_by_one_std_err(knn_tuner, desc(neighbors), metric = "roc_auc")

### Fit the final model -----------------------

oos_knn <- fit_resamples(
  finalize_workflow(wf_knn, k_1SE),
  resamples = folds_10, # Again - SAME resamples!!
  metrics = mset_class
)


# Compare models based on CV ------------------------------------------------



## Plot -----------------

cv_results <- bind_rows(
  logistic1 = collect_metrics(oos_logistic1, summarize = FALSE),
  logistic2 = collect_metrics(oos_logistic2, summarize = FALSE),
  KNN = collect_metrics(oos_knn, summarize = FALSE),
  .id = "model"
)

cv_summary <- bind_rows(
  logistic1 = collect_metrics(oos_logistic1),
  logistic2 = collect_metrics(oos_logistic2),
  KNN = collect_metrics(oos_knn),
  .id = "model"
)

chance_values <- data.frame(
  .metric = c("j_index", "roc_auc"),
  .estimate = c(0, 0.5)
)

cv_results |> 
  # Looking at ROC AUC
  ggplot(aes(model, .estimate)) + 
  facet_wrap(vars(.metric), scales = "free_y") + 
  # What is expected by chance alone
  geom_hline(aes(yintercept = .estimate),
             data = chance_values,
             linetype = "dashed") +
  # fold-data
  geom_line(aes(group = id)) + 
  # summary
  geom_pointrange(aes(y = mean, ymin = mean - std_err, ymax = mean + std_err),
                  data = cv_summary,
                  color = "red")



## Contrast ----------------

cv_compare_log1.knn <- cv_results |>
  pivot_wider(names_from = model, values_from = .estimate) |> 
  group_by(.metric) |> 
  summarise(
    # Because these are paired (fold-wise) we can do this:
    mean_diff = mean(logistic1 - KNN),
    se_diff = sd(logistic1 - KNN) / sqrt(n()),
    
    lb = mean_diff - 1.96 * se_diff,
    ub = mean_diff + 1.96 * se_diff
  )

cv_compare_log1.knn |> 
  # Format results:
  mutate(across(everything(), format)) |> 
  glue::glue_data(
    "A diff of {mean_diff} in {.metric}, 95% CI[{lb}, {ub}]"
  )


