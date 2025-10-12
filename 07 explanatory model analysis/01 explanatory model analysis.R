library(tidymodels)
# library(kknn)
# library(randomForest)

library(patchwork)

library(DALEX)
library(DALEXtra)
library(marginaleffects)


# Regression ------------------------------------------------------------------

## Fit a model ---------------------------------------------
# Let's fit a regression model!

data("Hitters", package = "ISLR")
?ISLR::Hitters

# split the data
set.seed(111)
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
Hitters.test <- testing(splits)

# fit a KNN with K=5
rec <- recipe(Salary ~ ., data = Hitters.train) |>
  step_naomit(Salary) |>
  step_dummy(all_factor_predictors()) |>
  step_interact(~ PutOuts:Walks) |>
  step_normalize(all_numeric_predictors())


knn_spec <- nearest_neighbor(
  mode = "regression",
  engine = "kknn",
  neighbors = 5
)

knn_fit <- workflow(
  preprocessor = rec,
  spec = knn_spec
) |>
  fit(data = Hitters.train)


## Explain the model -----------------------------------------------
# All the methods we'll be using are model agnostic - that means that they work
# for ANY model that can take ANY data and generate a prediction.
# See:
# https://www.tmwr.org/explain
# https://ema.drwhy.ai/

# We first need to setup an explainer:
knn_xplnr <- explain(
  knn_fit,
  label = "KNN (K=5)",
  # Pass the data that will be used to explain
  # WITHOUT the outcome variable...
  data = select(Hitters.train, -Salary),
  # ... which is passed separately
  y = Hitters.train$Salary
)


### Explain a single prediction ------------------------------
# Models make predictions. For example, we can see that our model predicts Bob
# Horner will have a salary of 1117 (*1000 = 1,117,000$)
predict(knn_fit, new_data = Hitters.test["-Bob Horner", ])


# But why?
# What is it about Bob that leads our model to make such a prediction?
mean(Hitters.train$Salary, na.rm = TRUE)


# We will be using SHAP (SHapley Additive exPlanations) to try and answer this
# question SHAP is a method that tries to estimate what each predictor
# contributes to each single prediction - accounting for whatever interactions
# or conditional effects it might have. The ideas are based on game theory -
# really interesting stuff. Lots of math.
#
# In essence, SHAP tries to attribute a predictions deviation from the mean
# prediction to the various predictors.

shap_bob <- predict_parts(
  knn_xplnr,
  new_observation = Hitters.test["-Bob Horner", ],
  type = "shap"
)

plot(shap_bob)
# - The contributions sum to the predicted values over the baseline (mean).
# - The box plots represent the distributions of contributions.
#   See: https://ema.drwhy.ai/shapley.html

plot(shap_bob, max_features = Inf) # get more features
plot(shap_bob, show_boxplots = FALSE) # show only mean SHAP values


# Note that predictions can be attributed differently between predictors
# across different predictions. Let's compare Bob to Willie:
shap_willie <- predict_parts(
  knn_xplnr,
  new_observation = Hitters.test["-Willie Wilson", ],
  type = "shap"
)

plot(shap_bob, max_features = Inf, show_boxplots = FALSE) +
  plot(shap_willie, max_features = Inf, show_boxplots = FALSE)


# Note that SHAP analysis DOES NOT give us any counterfactual information - we
# don't know that if Bob was in a different Division his salary would have been
# higher - just that the model chose to give him a lower prediction because he's
# in division W.
# In other words - we are explaining the model's predictions, but we are not
# explaining the salary! There is no causal information here, we are NOT in
# "explaining mode", we are still in "prediction mode"!

# Alternatives:
# - Local Interpretable Model-agnostic Explanations (LIME):
#   https://lime.data-imaginist.com
#   https://ema.drwhy.ai/LIME.html
#   These are useful for models with many (hundreds-thousands+) predictors
# - And more https://ema.drwhy.ai/InstanceLevelExploration.html

### Variable importance ----------------------
# We've already seen model-based variable importance methods, available through
# the {vip} package. But there are also model-agnostic methods, such as
# Permutation-based variable importance. See:
# https://ema.drwhy.ai/featureImportance.html

# This method assesses a predictors's contribution to a model's predictive
# accuracy by randomly shuffling a predictor's values - breaking its
# relationship with the outcome. The model's performance is then evaluated on
# this new permuted data.
# If the variable has no contribution, permuting it will have little to no
# effect on the model's performance, while variables with larger contributions
# will lead to larger and larger decrease in performance (e.g., larger RMSEs for
# regression).
# This process is repeated multiple times, and the average performance drop is
# used as the importance score, providing a robust measure of each variable's
# contribution.

vi_perm <- model_parts(
  knn_xplnr,
  B = 10, # Number of permutations
  variables = NULL # specify to only compute for some
)
plot(vi_perm, bar_width = 5)
# - The vertical line is the baseline RMSE
# - The horizontal bars are the increase in RMSE
# - Box plots are distribution of loss across permutations

plot(vi_perm, bar_width = 5, max_vars = Inf, show_boxplots = FALSE)


### Understand a variables contribution ---------

# Partial dependence plots (PDP) visualize the *marginal* effect of one or more
# predictor on the predicted outcome. That is, how the prediction is affected by
# changing the value of variable X while all other *are held constant*
# (_ceteris-paribus_). See:
#
# https://ema.drwhy.ai/partialDependenceProfiles.html
# https://marginaleffects.com/chapters/ml.html

pdp_hits <- model_profile(
  knn_xplnr,
  variables = "Years",
  # default is to plot results of 100 randomly sampled
  # observations.
  N = NULL
)
plot(pdp_hits) # average
# Note that this is a KNN model - it has no structure, and yet, this plot is
# fairly easy to understand!
plot(pdp_hits, geom = "profiles") # each line is a single outcome
plot(pdp_hits, geom = "points", variables = "Years")


# If you're not inserted in individual profiles, the {marginaleffects} package
# can also be used:
plot_predictions(
  knn_fit,
  by = "Years",
  # Define a counterfactual datagrid:
  newdata = datagrid(
    newdata = Hitters.train,
    grid_type = "counterfactual",

    Years = unique
  )
)


plot_predictions(
  knn_fit,
  by = "Division",
  # Define a counterfactual datagrid:
  newdata = datagrid(
    newdata = Hitters.train,
    grid_type = "counterfactual",

    Division = levels
  )
)
# Note that we don't have standard errors or confidence intervals.
# Just pure predictions - so these must be taken with a grain of salt.

plot_predictions(
  knn_fit,
  by = c("Years", "Division"),
  # Define a counterfactual datagrid:
  newdata = datagrid(
    newdata = Hitters.train,
    grid_type = "counterfactual",

    Years = unique,
    Division = levels
  )
)


plot_predictions(
  knn_fit,
  by = c("Walks", "Years"),
  # Define a counterfactual datagrid:
  newdata = datagrid(
    newdata = Hitters.train,
    grid_type = "counterfactual",

    Walks = unique,
    Years = \(v) as.integer(mean(v) + c(-1, 0, 1) * sd(v))
  )
)


# Classification ------------------------------------------------------------
# Let's apply all these methods to a (multi-class, probabilistic) classification
# model.

## Fit a model -------------------------------------
# Let's fit a classification model!

data("penguins", package = "palmerpenguins")
?palmerpenguins::penguins

# split the data
set.seed(111)
splits <- initial_split(penguins, prop = 0.7)
penguins.train <- training(splits)
penguins.test <- testing(splits)

# We'll fit a random forest model:
rec <- recipe(species ~ ., data = penguins.train) |>
  step_rm(year, island) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_impute_mode(all_factor_predictors())

rf_spec <- rand_forest(
  mode = "classification",
  engine = "randomForest",
  mtry = sqrt(.cols() - 1),
  trees = 500,
  min_n = 5
)

rf_fit <-
  workflow(preprocessor = rec, spec = rf_spec) |>
  fit(data = penguins.train)


## Explain the model -----------------------------------------------

rf_xplnr <- explain(
  rf_fit,
  label = "Random Forest",
  data = select(penguins.train, -species),
  y = penguins.train$species
)


### Explain a single prediction ------------------------------
# Why does the model think that obs 61 has a high chance of being a Gentoo?
predict(rf_fit, new_data = penguins.test[61, ], type = "prob")


# We can look at his SHAP values:
shap_61 <- predict_parts(
  rf_xplnr,
  new_observation = penguins.test[61, ],
  type = "shap"
)
plot(shap_61)
# We get the SHAP values for each class!

### Variable importance ----------------------

# Since we're using a random forest model, we can get model-based variable
# importance:
extract_fit_engine(rf_fit) |>
  vip::vi_model() |>
  vip::vip(num_features = 100)


# But we can still use the permutation method (or a multiclass model, the loss
# function is a measure of entropy):
vi_perm <- model_parts(rf_xplnr, B = 10, variables = NULL)
plot(vi_perm, max_vars = Inf)
# This matches the plot above very well!
# What's going on with year / island?

### Understand a variables contribution ---------

model_profile(rf_xplnr, variables = c("bill_length_mm", "body_mass_g")) |>
  plot() +
  coord_cartesian(ylim = c(0, 1))
# As far as bill length goes, it seems like smaller bills predict Adelie, larger
# for Chinstrap, with Gentoo somewhere in the middle!

# Let's see if this makes sense...
ggplot(penguins.train, aes(bill_length_mm, body_mass_g, color = species)) +
  geom_point()


plot_predictions(
  rf_fit,
  type = "prob",
  by = c("body_mass_g", "bill_length_mm", "group"), # group = class
  # Define a counterfactual datagrid:
  newdata = datagrid(
    newdata = penguins.train,
    grid_type = "counterfactual",

    body_mass_g = unique,
    bill_length_mm = \(v) range(v, na.rm = TRUE)
  )
) +
  coord_cartesian(ylim = c(0, 1))
