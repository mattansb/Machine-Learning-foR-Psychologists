library(patchwork)

library(tidymodels)
library(tailor)
# library(kknn)

mirai::daemons(4) # Use 4 CPU cores for parallel processing

# The Data & Problem --------------------------------------------------------

# Let's look at
data(ad_data, package = "modeldata")
?modeldata::ad_data

levels(ad_data$Class)
# (No need to relevel the factor - the *first* class is the event class.)

# Data Splitting -
set.seed(20251201)
splits <- initial_split(ad_data, prop = 0.7, strata = Class)
Alz.train <- training(splits)
Alz.test <- testing(splits)

mset_class <- metric_set(sensitivity, specificity, accuracy, j_index, roc_auc)
# J index = sensitivity + specificity - 1

# Fit a model -------------------------------------------------------------

# Preprocessing
rec <- recipe(Class ~ ., data = Alz.train) |>
  step_lincomb(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors())

# Model Spec
knn_spec <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  neighbors = 15 # arbitrary choice for now
)

knn_wf1 <- workflow(preprocessor = rec, spec = knn_spec)

knn_fit1 <- fit(knn_wf1, data = Alz.train)


## Evaluate on hold-out set ---------------------------------

Alz.test_pred0.50 <- augment(knn_fit1, new_data = Alz.test)
Alz.test_pred0.50 |>
  mset_class(truth = Class, estimate = .pred_class, .pred_Impaired)
# The model is pretty good, but early detection of Alzheimer's disease is
# important - we want higher sensitivity (at the cost of lower specificity).
# (Note also that the data is imbalanced, but even so...)

# What can we do?

# Find a threshold -----------------------------------------------------------

# To change the threshold, we introduce a new type of step in the workflow:
# post-processing.
tlr <- tailor() |>
  adjust_probability_threshold(threshold = tune())

# There are many more adjustments that can be made:
# https://tailor.tidymodels.org/reference/index.html
# https://probably.tidymodels.org/reference/index.html
# e.g., for calibration.
?probably::cal_plot_windowed
?adjust_probability_calibration


# Bring is all together in a workflow
knn_wf2 <- workflow(preprocessor = rec, spec = knn_spec, postprocessor = tlr)
knn_wf2

## Tune the threshold -------------------------------------------------

tune_results <- tune_grid(
  knn_wf2,
  resamples = vfold_cv(Alz.train, v = 10, strata = Class),
  grid = tibble(threshold = seq(0, 1, length = 20)),
  metrics = mset_class,
)

# As expected, we can see that sensitivity and specificity trade off from 0/1 to
# 1/0 as we change the threshold, and J-index seems to hit a good balance around 0.3.
autoplot(tune_results, metric = c("sensitivity", "specificity", "j_index"))

# Accuracy is also affected by threshold changes - but AUC is *not*. Why?
autoplot(tune_results, metric = c("accuracy", "roc_auc"))

## Finalize model ------------------------------------------------

knn_fit2 <- knn_wf2 |>
  finalize_workflow(
    # select the best threshold based on J-index measure
    parameters = select_best(tune_results, metric = "j_index")
  ) |>
  fit(data = Alz.train)

# We can also select by optimizing multiple metrics simultaneously with the
# {desirability} package:
# https://.tidymodels.org/#using-with-the-tune-package
?desirability2::select_best_desirability()


knn_fit2


# Predict (+ adjust) on test set ---------------------------------

Alz.test_pred0.30 <- augment(knn_fit2, new_data = Alz.test)

# We can see that the class predictions do not completely agree:
table(
  "Default" = Alz.test_pred0.50$.pred_class,
  "Adjusted" = Alz.test_pred0.30$.pred_class
)

# (But note that probabilistic metrics are unaffected by adjustment)
plot(
  Alz.test_pred0.50$.pred_Impaired,
  Alz.test_pred0.30$.pred_Impaired,
  xlab = "Default (thresh = 0.50)",
  ylab = "Adjusted (thresh = 0.30)"
)

Alz.test_pred0.50 |>
  mset_class(truth = Class, estimate = .pred_class, .pred_Impaired)

Alz.test_pred0.30 |>
  mset_class(truth = Class, estimate = .pred_class, .pred_Impaired)
