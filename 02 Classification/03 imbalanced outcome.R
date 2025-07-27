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


# The worst model ---------------------------------------------------------
# The data imbalance means that, technically, we can achieve high accuracy by
# simply predicting "No"....

Caravan.test$.pred_class.BAD <- factor("No", levels = c("Yes", "No"))
Caravan.test$.pred_Yes.BAD <- 0

Caravan.test |> conf_mat(truth = Purchase, estimate = .pred_class.BAD)


# But as we can see, we have no specificity!
mset_class <- metric_set(accuracy, specificity, sensitivity)
Caravan.test |> mset_class(truth = Purchase, estimate = .pred_class.BAD)


# Training with Imbalance Data --------------------------------------------

rec <- recipe(Purchase ~ ., data = Caravan.train) |>
  step_normalize(all_numeric_predictors())


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
Caravan.test_predictions <- augment(knn_fit, new_data = Caravan.test)
Caravan.test_up.predictions <- augment(knn_fit.up, new_data = Caravan.test)
Caravan.test_down.predictions <- augment(knn_fit.down, new_data = Caravan.test)


Caravan.test |>
  mset_class(truth = Purchase, estimate = .pred_class.BAD)
Caravan.test_predictions |>
  mset_class(truth = Purchase, estimate = .pred_class)
Caravan.test_up.predictions |>
  mset_class(truth = Purchase, estimate = .pred_class)
Caravan.test_down.predictions |>
  mset_class(truth = Purchase, estimate = .pred_class)

# As we can see, the accuracy (and specificity) have dropped, but sensitivity is
# higher.

# We can also compare ROC curves and AUCs:
ROCs <- bind_rows(
  Fixed = Caravan.test |> roc_curve(truth = Purchase, .pred_Yes.BAD),
  None = Caravan.test_predictions |> roc_curve(truth = Purchase, .pred_Yes),
  Up = Caravan.test_up.predictions |> roc_curve(truth = Purchase, .pred_Yes),
  Down = Caravan.test_down.predictions |>
    roc_curve(truth = Purchase, .pred_Yes),

  .id = "Method"
)

autoplot(ROCs) + aes(color = Method)
