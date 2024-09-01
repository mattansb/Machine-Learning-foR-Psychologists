
library(tidymodels)
# library(kknn)

# The Data & Problem --------------------------------------------------------

data("Caravan", package = "ISLR")
?ISLR::Caravan

levels(Caravan$Purchase)
# We will relevel the factor so the *first* class is the event class!
Caravan$Purchase <- relevel(Caravan$Purchase, ref = "Yes")
levels(Caravan$Purchase)
# This way we won't have to set `event_level = "second"` everywhere.

# Data Splitting
set.seed(1234)
splits <- initial_split(Caravan, prop = 0.7)
Caravan.train <- training(splits)
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

Caravan.test_predictions <- augment(knn_fit, new_data = Caravan.test)
head(Caravan.test_predictions)

mset_class <- metric_set(sensitivity, specificity, accuracy)
thresholds <- seq(0, 1, length = 20)

threshold_metrics <- Caravan.test_predictions |> 
  probably::threshold_perf(truth = Purchase, estimate = .pred_Yes,
                           thresholds = thresholds, 
                           metrics = mset_class)

# Plot metrics vs threshold
ggplot(threshold_metrics, aes(x = .threshold, y = .estimate, color = .metric)) +
  geom_line() +
  facet_grid(rows = vars(.metric)) +
  theme_bw() +
  labs(y = "Metric Value", title = "Metrics vs Threshold")



# Set threshold -----------------------------------

Caravan.test_predictions$.pred_class0.1 <- 
  # this function is defined with respect to the FIRST level
  probably::make_two_class_pred(
    estimate = Caravan.test_predictions$.pred_Yes,
    levels = levels(Caravan.test_predictions$Purchase),
    threshold = 0.1 
  )

Caravan.test_predictions |> 
  mset_class(truth = Purchase, estimate = .pred_class)
Caravan.test_predictions |> 
  mset_class(truth = Purchase, estimate = .pred_class0.1)



# re-threshold metrics ----------------------------------------------------

source(".rethresh.R")

accuracy0.1 <- metric_tweak("accuracy0.1", .fn = rethresh,
                            threshold = 0.1,
                            class_metric = accuracy_vec)

Caravan.test_predictions |> accuracy(Purchase, estimate = .pred_class)
Caravan.test_predictions |> accuracy0.1(Purchase, .pred_Yes)
# same as:
Caravan.test_predictions |> accuracy(Purchase, estimate = .pred_class0.1)

