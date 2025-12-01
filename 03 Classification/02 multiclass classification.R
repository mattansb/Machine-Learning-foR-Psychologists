library(tidymodels)
# library(kknn)

# This script demonstrates how we might asses a multiclass-classification model.

# Data and problem ----------------------------------------------------------

data("penguins", package = "palmerpenguins")
?palmerpenguins::penguins
# This data set contains info on penguins from the Palmer Archipelago,
# Antarctica. We will predict the species of penguins based on their bill length
# and depth, using a KNN model.

set.seed(20251201)
splits <- initial_split(penguins, prop = 0.8, strata = species)
penguins.train <- training(splits)
penguins.test <- testing(splits)


# Fit ---------------------------------------------------------------------

knn_spec <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  neighbors = 5
)

rec <- recipe(species ~ body_mass_g + sex, data = penguins.train) |>
  # The data contains missing values - we will impute using the median / mode:
  step_impute_median(body_mass_g) |>
  step_impute_mode(sex) |>
  # We need to dummy code the factor predictors for KNN
  step_dummy(sex) |>
  # We need to normalize the predictors for KNN
  step_normalize(all_numeric_predictors())


knn_wf <- workflow(preprocessor = rec, spec = knn_spec)

knn_fit <- fit(knn_wf, data = penguins.train)


# Predict -----------------------------------------------------------------

penguins.test_predictions <- augment(knn_fit, new_data = penguins.test)
head(penguins.test_predictions)

# Unlike binary classification, we don't typically have a "positive" class, so
# we can't really compute sensitivity, specificity, etc. Instead, we can use
# several methods to summarize the performance of a multiclass model.
#
# Note, however that accuracy does not require a "positive" class, and so it can
# be used without issue in multiclass problems.
mset_class <- metric_set(sensitivity, specificity, accuracy)


# {yardstick} provids 3 methods for dealing with multiclass predictions, all of
# them effectively compute such metrics for each class, and them average them.
# You can read about these here:
# https://yardstick.tidymodels.org/articles/multiclass.html

## Macro (standard averaging)
penguins.test_predictions |>
  mset_class(truth = species, estimate = .pred_class, estimator = "macro")

## Weighted Macro (weight by class frequancy)
penguins.test_predictions |>
  mset_class(
    truth = species,
    estimate = .pred_class,
    estimator = "macro_weighted"
  )

## Micro (sort of observation wise averaging)
penguins.test_predictions |>
  mset_class(truth = species, estimate = .pred_class, estimator = "micro")


# I also provide here a function to produce event-wise - not averaging!
source(".metric_by_event.R")

penguins.test_predictions |>
  metric_by_event(mset_class, truth = species, estimate = .pred_class)
# We can see we have great sensitivity and specificity Gentoo, while having poor
# sensitivity for Chinstrap / specificity for Adelie. This is because the model
# is very good at predicting Gentoo, but not so good at predicting the other
# classes.

# We can get a sense for this using a ROC curve:
penguins.test_predictions |>
  roc_curve(truth = species, .pred_Adelie, .pred_Chinstrap, .pred_Gentoo) |>
  autoplot() +
  aes(color = .level)

# We can get an "average" AUC (using Hand & Till method):
penguins.test_predictions |>
  roc_auc(truth = species, .pred_Adelie, .pred_Chinstrap, .pred_Gentoo)
