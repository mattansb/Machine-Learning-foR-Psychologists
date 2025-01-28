
library(tidymodels)
# library(kknn)
library(tailor)

# The Data & Problem --------------------------------------------------------

data("Caravan", package = "ISLR")
?ISLR::Caravan

levels(Caravan$Purchase)
# We will relevel the factor so the *first* class is the event class!
Caravan$Purchase <- relevel(Caravan$Purchase, ref = "Yes")
levels(Caravan$Purchase)
# This way we won't have to set `event_level = "second"` everywhere.


# Data Splitting -
# we will need a seperate validation set to "tune" the threshold.
set.seed(1234)
splits <- initial_validation_split(Caravan, prop = c(0.6, 0.2), strata = Purchase)
Caravan.train <- training(splits)
Caravan.valid <- validation(splits)
Caravan.test <- testing(splits)



# Fit a model -------------------------------------------------------------

rec <- recipe(Purchase ~ ., 
              data = Caravan.train) |> 
  step_normalize(all_numeric_predictors())

knn_spec <- nearest_neighbor(
  mode = "classification", engine = "kknn",
  neighbors = 10
)

knn_wf <- workflow(preprocessor = rec, spec = knn_spec)

knn_fit <- fit(knn_wf, data = Caravan.train)




# Find threshold -----------------------------------------------------------
# We will find the best threshold by using the hold out validation set.


mset_class <- metric_set(sensitivity, specificity, accuracy, f_meas)


Caravan.valid_predictions <- augment(knn_fit, new_data = Caravan.valid)
head(Caravan.valid_predictions)


threshold_metrics <- Caravan.valid_predictions |> 
  probably::threshold_perf(
    truth = Purchase, 
    estimate = .pred_Yes,
    
    thresholds = seq(0, 1, length = 20), 
    
    metrics = mset_class
  )


# Plot metrics vs threshold
ggplot(threshold_metrics, aes(x = .threshold, y = .estimate, color = .metric)) +
  facet_grid(rows = vars(.metric), scales = "free_y") +
  geom_point(size = 3, 
             data = \(d) slice_max(d, .estimate, by = .metric)) +
  geom_line(linewidth = 1) +
  theme_bw() +
  labs(y = "Metric Value", title = "Metrics vs Threshold")

# Note that we already saw that the classes are imbalanced:
table(Caravan.valid$Purchase)


## Select threshold -----------------------------------

# Find the threshold that maximizes f_meas
th <- threshold_metrics |> 
  filter(.metric == "f_meas") |> 
  slice_max(.estimate) |> 
  pull(.threshold)
th

# Make a post-processing model adjuster
tlr <- tailor() |> 
  adjust_probability_threshold(threshold = th) |> 
  fit(
    .data = Caravan.valid_predictions, 
    outcome = Purchase, 
    estimate = .pred_class,
    probabilities = c(.pred_Yes, .pred_No)
  )
# This post-processing fit can be used to apply the new threshold




# Predict (+ adjust) on test set ---------------------------------


Caravan.test_pred0.50 <- augment(knn_fit, new_data = Caravan.test)
Caravan.test_pred0.11 <- predict(tlr, new_data = Caravan.test_pred0.50)

# We can see that the class predictions do not completely agree:
table(
  "Default" = Caravan.test_pred0.50$.pred_class,
  "Adjusted" = Caravan.test_pred0.11$.pred_class
)


Caravan.test_pred0.50 |> 
  mset_class(truth = Purchase, estimate = .pred_class)

Caravan.test_pred0.11 |> 
  mset_class(truth = Purchase, estimate = .pred_class)
# The adjusted threshold gives a better sens/spec balance.
