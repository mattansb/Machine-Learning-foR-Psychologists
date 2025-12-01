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
splits <- initial_split(Caravan, prop = 0.7, strata = Purchase)
Caravan.train <- training(splits)
Caravan.test <- testing(splits)


table(Caravan.train$Purchase) |> proportions()
# As we can see, the classes are very unbalanced.

# Note we used a stratified split, so that both the train and test set have
# about the ~same distribution of classes. This is particularly important with
# imbalanced data.
table(Caravan.test$Purchase) |> proportions()


# We will be using this recipe:
rec <- recipe(Purchase ~ ., data = Caravan.train) |>
  step_normalize(all_numeric_predictors())

# And these classification metrics:
mset_class <- metric_set(accuracy, specificity, sensitivity)


# The worst model ---------------------------------------------------------
# The data imbalance means that, technically, we can achieve high accuracy by
# simply predicting "No"....

null_fit <-
  workflow(preprocessor = rec, spec = null_model(mode = "classification")) |>
  fit(data = Caravan.train)
# The null model is the worst possible model - it does not use any information
# in X, only the distribution of Y in the training set.
# - For regression, it always predicts mean(Y)
# - For classification, it predicts the frequent class and base rate
#   probabilities.
# These models are good for benchmarking.

# But like a "real" model, it can be used to generate predictions on new data:
Caravan.test_predictions_NULL <- augment(null_fit, new_data = Caravan.test)

# Same as:
# Caravan.test$.pred_class.BAD <- factor("No", levels = c("Yes", "No"))
# Caravan.test$.pred_Yes.BAD <- 0

Caravan.test_predictions_NULL |>
  conf_mat(truth = Purchase, estimate = .pred_class)

# But as we can see, we have no specificity!
Caravan.test_predictions_NULL |>
  mset_class(truth = Purchase, estimate = .pred_class)


# Training with Imbalance Data --------------------------------------------

knn_spec <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  neighbors = 10
)

knn_wf <- workflow(preprocessor = rec, spec = knn_spec)

knn_fit <- fit(knn_wf, data = Caravan.train)


# Up- and Down-Sampling ---------------------------------------------------

# We can also sample our data such that we artificially achieve class balances.
# The main methods are:
# down-sampling:
#   randomly subset all the classes in the training set so that their class
#   frequencies match the least prevalent class.
# up-sampling:
#   randomly sample (with replacement) the minority class(es) to be the same
#   size as the majority class.
# hybrid methods:
#   techniques such as SMOTE and ROSE down-sample the majority class and
#   synthesize new data points in the minority class.
# See https://themis.tidymodels.org/

# We will use up- and down-sampling:

rec_up <- rec |> themis::step_upsample(Purchase)
rec_down <- rec |> themis::step_downsample(Purchase)

knn_fit.up <- knn_wf |>
  update_recipe(rec_up) |>
  fit(data = Caravan.train)
knn_fit.down <- knn_wf |>
  update_recipe(rec_down) |>
  fit(data = Caravan.train)


# Comparing Results -------------------------------------------------------

# Get raw predictions
Caravan.test_predictions <-
  bind_rows(
    "NULL" = Caravan.test_predictions_NULL,
    None = augment(knn_fit, new_data = Caravan.test),
    Up = augment(knn_fit.up, new_data = Caravan.test),
    Down = augment(knn_fit.down, new_data = Caravan.test),

    .id = "Method"
  ) |>
  mutate(
    Method = factor(Method, levels = c("NULL", "None", "Up", "Down"))
  )

# Let's look at the predictions made by the different methods for the first test
# case:
Caravan.test_predictions |>
  select(Method, Purchase, starts_with(".pred")) |>
  slice(1, .by = Method)


Caravan.test_predictions |>
  group_by(Method) |>
  mset_class(truth = Purchase, estimate = .pred_class)
# As we can see, the accuracy (and specificity) have dropped, but sensitivity is
# higher.

# We can also compare ROC curves and AUCs:
Caravan.test_predictions |>
  group_by(Method) |>
  roc_curve(truth = Purchase, .pred_Yes) |>
  autoplot()
