library(patchwork)

library(tidymodels)
library(probably)
library(tailor)
# library(kknn)

# The Data & Problem --------------------------------------------------------

# Let's look at
data(ad_data, package = "modeldata")
?modeldata::ad_data

levels(ad_data$Class)
# (No need to relevel the factor - the *first* class is the event class.)

# Data Splitting -
set.seed(20251201)
splits <- initial_validation_split(ad_data, prop = c(0.6, 0.2), strata = Class)
Alz.train <- training(splits)
Alz.valid <- validation(splits)
Alz.test <- testing(splits)

mset_class <- metric_set(sensitivity, specificity, accuracy, f_meas)


# Fit a model -------------------------------------------------------------

rec <- recipe(Class ~ ., data = Alz.train) |>
  step_lincomb(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors())

knn_spec <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  neighbors = 15
)

knn_wf <- workflow(preprocessor = rec, spec = knn_spec)

knn_fit <- fit(knn_wf, data = Alz.train)


Alz.valid_predictions <- augment(knn_fit, new_data = Alz.valid)
Alz.valid_predictions |>
  mset_class(truth = Class, estimate = .pred_class, .pred_Impaired)
# The model is pretty good, but early detection of Alzheimer's disease is
# important - we want higher sensitivity (at the cost of lower specificity).
# (Note also that the data is imbalanced, but even so...)

# What can we do?

# Find threshold -----------------------------------------------------------
# We will find the best threshold by using the hold out validation set.

threshold_metrics <- Alz.valid_predictions |>
  threshold_perf(
    truth = Class,
    estimate = .pred_Impaired,

    thresholds = seq(0, 1, length = 20),

    metrics = mset_class
  )

p_metrics <- threshold_metrics |>
  filter(.metric != "f_meas") |>
  ggplot(aes(x = .threshold, y = .estimate, color = .metric)) +
  geom_point(size = 3, data = \(d) slice_max(d, .estimate, by = .metric)) +
  geom_line(aes(linetype = .metric == "accuracy"), linewidth = 1) +
  geom_vline(xintercept = 0.5) +
  theme_bw() +
  labs(
    y = "Metric Value",
    title = "Metrics vs Threshold"
  ) +
  guides(linetype = "none")

p_f1 <-
  threshold_metrics |>
  filter(.metric == "f_meas") |>
  ggplot(aes(x = .threshold, y = .estimate)) +
  geom_point(size = 3, data = \(d) slice_max(d, .estimate, by = .metric)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = 0.5) +
  theme_bw() +
  labs(y = "F1", title = "Metrics vs Threshold")


p_metrics / p_f1 + plot_layout(heights = c(2, 1))
# Looks like a threshold of about 0.30 will give us a high sens without too much
# loss of spec.

## Select threshold -----------------------------------

# We introduce a new type of step in our workflow: POST-processing.
# This is a step that adjusts predictions. There are many types of adjustments...
tlr <- tailor() |>
  # ... we want to adjust the binary classifications threshold.
  adjust_probability_threshold(threshold = 0.30)

# There are many more adjustments that can be made:
# https://tailor.tidymodels.org/reference/index.html
# https://probably.tidymodels.org/reference/index.html
# e.g., for calibration.
?cal_plot_windowed
?adjust_probability_calibration


# We can add this step to out workflow:
knn_wf2 <- workflow(
  preprocessor = rec,
  spec = knn_spec,
  postprocessor = tlr
)

knn_wf2

knn_fit2 <- fit(knn_wf2, data = Alz.train)


## Predict (+ adjust) on test set ---------------------------------

Alz.test_pred0.50 <- augment(knn_fit, new_data = Alz.test)
Alz.test_pred0.30 <- augment(knn_fit2, new_data = Alz.test)

# We can see that the class predictions do not completely agree:
table(
  "Default" = Alz.test_pred0.50$.pred_class,
  "Adjusted" = Alz.test_pred0.30$.pred_class
)

Alz.test_pred0.50 |> mset_class(truth = Class, estimate = .pred_class)
Alz.test_pred0.30 |> mset_class(truth = Class, estimate = .pred_class)

# (Note that probabilistic metrics are unaffected by adjustment)
plot(Alz.test_pred0.50$.pred_Impaired, Alz.test_pred0.30$.pred_Impaired)
Alz.test_pred0.50 |> roc_auc(truth = Class, .pred_Impaired)
Alz.test_pred0.30 |> roc_auc(truth = Class, .pred_Impaired)
