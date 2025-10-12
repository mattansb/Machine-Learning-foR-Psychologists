library(tidymodels)
# library(randomForest)

library(patchwork)

library(DALEX)
library(DALEXtra)


# Fit a model ---------------------------------------------
# Let's fit a regression model!

data("Hitters", package = "ISLR")
?ISLR::Hitters

# split the data
set.seed(111)
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
Hitters.test <- testing(splits)

# fit an RF with mtry = 7
rec <- recipe(Salary ~ ., data = Hitters.train) |>
  step_naomit(Salary) |>
  step_center(Hits, Years, HmRun) |>
  step_interact(~ Hits:Years) |>
  step_poly(HmRun, degree = 2)


rf_spec <- rand_forest(
  mode = "regression",
  engine = "randomForest",
  mtry = 7,
  trees = 500,
  min_n = 5
)

rf_fit <- workflow(
  preprocessor = rec,
  spec = rf_spec
) |>
  fit(data = Hitters.train)


# Explain predictors  -----------------------------------------------
# This is the default behavior:
# 1. Input to explainer: raw data
# 2. Model builds features and make predictons

rf_xpln_predictors <- explain(
  rf_fit,
  label = "predictors",
  # Pass the data that will be used to explain
  # WITHOUT the outcome variable...
  data = select(Hitters.train, -Salary),
  # ... which is passed separately
  y = Hitters.train$Salary
)

## Explain a single prediction ------------------------------
# Let's explain bob again:
predict(rf_fit, new_data = Hitters.test["-Bob Horner", ])


shap_bob_predictors <- predict_parts(
  rf_xpln_predictors,
  new_observation = Hitters.test["-Bob Horner", ],
  type = "shap"
)


### Variable importance ----------------------

vi_perm_predictors <- model_parts(
  rf_xpln_predictors,
  B = 10, # Number of permutations
  variables = NULL
)


# Explain features  -----------------------------------------------

# But we can also explain the features:
rf_rec_trained <- extract_recipe(rf_fit)
rf_spec_trained <- extract_fit_parsnip(rf_fit)

rf_xpln_features <- explain(
  rf_spec_trained,
  label = "features",

  # Pass the processed data that will be used to explain
  data = bake(
    rf_rec_trained,
    new_data = Hitters.train,
    # WITHOUT the outcome variable...
    -all_outcomes()
  ),

  # ... which is passed separately
  y = Hitters.train$Salary
)

Hitters.test_baked <- bake(rf_rec_trained, new_data = Hitters.test)


## Explain a single prediction ------------------------------
# Let's explain bob again - note that these are (and must!) be the same:
predict(
  rf_fit,
  new_data = Hitters.test["-Bob Horner", ]
)

predict(
  rf_spec_trained,
  new_data = Hitters.test_baked[rownames(Hitters.test) == "-Bob Horner", ]
)


shap_bob_features <- predict_parts(
  rf_xpln_features,
  new_observation = Hitters.test_baked[
    rownames(Hitters.test) == "-Bob Horner",
  ],
  type = "shap"
)

plot(shap_bob_predictors, max_features = Inf, show_boxplots = FALSE) +
  plot(shap_bob_features, max_features = Inf, show_boxplots = FALSE)
# Note HmRun_poly_2 and Hits_x_Years are features, not predictors!

### Variable importance ----------------------

# Note that permutation of features might not make sense in some cases - for
# example, HmRun and HmRun^2 are dependent features, as are Hits_x_Years and
# Hits / Years so permuting one but not the others might lead to unrealistic
# values.
vi_perm_features <- model_parts(
  rf_xpln_features,
  B = 10, # Number of permutations
  variables = NULL
)


plot(vi_perm_predictors, bar_width = 5) +
  plot(vi_perm_features, bar_width = 5)
