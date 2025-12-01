library(tidymodels)
# library(kernlab)

# The OJ Data -----------------------------------------------------------------

data("OJ", package = "ISLR")
?ISLR::OJ
# Which brand of orange juice was purchased?

# We don't really "care" about the "event", so we will not use any
# "even"-related metrics. This also means the order of the classes does not
# mater
levels(OJ$Purchase)
oj_metrics <- metric_set(accuracy, f_meas, roc_auc)


set.seed(1)
splits <- initial_split(OJ, prop = 0.8)
OJ.train <- training(splits)
OJ.test <- testing(splits)

ggplot(OJ.train, aes(PriceDiff, LoyalCH, color = Purchase)) +
  geom_point()
# Looks people are more likely to but CH if:
# - MM is more expensive
# - They have brand loyalty

# We will test this:
rec <- recipe(Purchase ~ PriceDiff + LoyalCH, data = OJ.train) |>
  step_normalize(all_numeric_predictors())
# The decision boundary maximizes the DISTANCE to the nearest data points from
# different classes. Hence, the distance between data points affects the
# decision boundary SVM chooses. In other words, training an SVM over the scaled
# and non-scaled data leads to the generation of different models.
#
# Note: We're only using 2 predictors because it allowed us to visualize the
# "space". But SVM can of course use many predictors.

folds <- vfold_cv(OJ.train, v = 5)


# Support Vector Classifier --------------------------------------------------
# FITTING the support vector classifier

## Tune ------------------------

# We will fit the support vector classifier by tuning the C hyperparameter.
# For annoying reasons, we actually tune a "cost" parameter -
#               WHICH IS INVERSLY RELATED TO C!
# It really should be called a "penalty", since it allows us to specify the
# penalty of a violation to the margin, such that:
#   Small cost -> wide margins and many support vectors.
#   Large cost -> narrow margins and few support vectors.
#
# Here, we directly try more then one value and find the best fit with CV!
svmlin_spec <- svm_linear("classification", engine = "kernlab", cost = tune())

translate(svmlin_spec)
# Note that this engine converts the SVM to a probabilistic classifier
# using Platt scaling (a logistic regression model is fit to the SVM output).

svmlin_wf <- workflow(preprocessor = rec, spec = svmlin_spec)

# Tune the cost between being very low and very large:
(svmlin_grid <- grid_regular(cost(), levels = 20))


svmlin_tune <- tune_grid(
  svmlin_wf,
  resamples = folds,
  grid = svmlin_grid,
  metrics = oj_metrics
)

autoplot(svmlin_tune)

collect_metrics(svmlin_tune) |>
  filter(.metric == "f_meas")

(svmlin_const <- select_best(svmlin_tune, metric = "f_meas"))

## The final model ------------------------------------------------------

svmlin_fit <- svmlin_wf |>
  finalize_workflow(svmlin_const) |>
  fit(data = OJ.train)


# We can explore the model but extracting the underlying model object:
(svmlin_eng <- extract_fit_engine(svmlin_fit))
# training error is 0.17 (for the cost 0.13)
# Number of Support Vectors : 374!

# Here we demonstrated the use of this function on a two-dimensional example so
# that we can plot the resulting decision boundary.
X_train <- bake(
  extract_recipe(svmlin_fit),
  new_data = OJ.train,
  all_predictors()
)

plot(svmlin_eng, data = X_train)
# Full circles\triangles = support vectors (374 obs.)
# Hollow circles\triangles = the remaining observations

## Evaluating the support vector classifier performance ---------------------
# (on a new TEST DATA)

# Predict the class labels of these test observations.
svmlin_predictions <- augment(svmlin_fit, new_data = OJ.test)
svmlin_predictions |> conf_mat(Purchase, .pred_class)
svmlin_predictions |> oj_metrics(Purchase, estimate = .pred_class, .pred_CH)


# Support Vector Machine ----------------------------------------
# (non-linear kernel)

## Polynomial ------------------------------------

svmpoly_spec <- svm_poly(
  mode = "classification",
  engine = "kernlab",

  cost = tune(),
  degree = 3,
  scale_factor = 1
)
# Here we also add two new arguments (that can be tuned):
# - degree: specifies the polynomial degree (e.g., quadratic: degree = 2).
# - scale_factor: can be used to re-scale the data, but it is better to
#   pre-standardized the data and set this to 1.

translate(svmpoly_spec)

svmpoly_wf <- workflow(preprocessor = rec, spec = svmpoly_spec)


svmpoly_tune <- tune_grid(
  svmpoly_wf,
  resamples = folds,
  grid = svmlin_grid, # reuse cost grid
  metrics = oj_metrics
)

(svmpoly_params <- select_best(svmpoly_tune, metric = "f_meas"))


svmpoly_fit <- svmpoly_wf |>
  finalize_workflow(svmpoly_params) |>
  fit(data = OJ.train)


(svmpoly_eng <- extract_fit_engine(svmpoly_fit))
# best training error is 0.17 (for the cost 0.045)
# Number of Support Vectors : 359!
plot(svmpoly_eng, data = X_train)


## Radial ------------------------------------

# Here we also add the sigma argument to specify a value of sigma for the radial
# basis kernel. It can be thought of as the 'spread' of the decision region.
# When sigma is low, the 'curve' of the decision boundary is very low and thus
# the decision region is very broad. Intuitively, the sigma parameter defines
# how far the influence of a single training example reaches, with low values
# meaning ‘far’ and high values meaning ‘close’. (this is related to the inverse
# of the sigma parameter of a normal distribution, but we will not dig into
# that... ) nice post I ran into:
# https://vitalflux.com/svm-rbf-kernel-parameters-code-sample/

svmrad_spec <- svm_rbf(
  "classification",
  engine = "kernlab",
  cost = 0.1, # we get it...
  rbf_sigma = 2
)
# rbf_sigma is the sigma parameter for the radial basis kernel, which controls
# the smoothness of the decision boundary.

translate(svmrad_spec)

svmrad_wf <- workflow(preprocessor = rec, spec = svmrad_spec)

svmrad_fit <- fit(svmrad_wf, data = OJ.train)


(svmrad_eng <- extract_fit_engine(svmrad_fit))
# best training error is 0.17 (for the cost 0.1)
# Number of Support Vectors : 469!
plot(svmrad_eng, data = X_train)


## Compare models ----------------------------------------------------------

svmpoly_predictions <- augment(svmpoly_fit, new_data = OJ.test)
svmrad_predictions <- augment(svmrad_fit, new_data = OJ.test)


svmlin_predictions |> oj_metrics(Purchase, estimate = .pred_class, .pred_CH)
svmpoly_predictions |> oj_metrics(Purchase, estimate = .pred_class, .pred_CH)
svmrad_predictions |> oj_metrics(Purchase, estimate = .pred_class, .pred_CH)
# These are all basically the same...

# Multi-classes SVM -------------------------------------------------

data("penguins", package = "palmerpenguins")
?palmerpenguins::penguins

set.seed(1)
splits <- initial_split(penguins, prop = 0.7)
penguins.train <- training(splits)
penguins.test <- testing(splits)


# We will try to predict penguins species from bill_length_mm and body_mass_g.

table(penguins.train$species)
table(penguins.train$species) |> proportions()
# If the response is a factor containing more than two levels, then fit() will
# perform multi-class classification using the one-versus-one approach.

svmlin_spec2 <- svm_linear("classification", engine = "kernlab", cost = 0.1)

rec2 <- recipe(species ~ bill_length_mm + body_mass_g, data = penguins.train) |>
  step_normalize(all_numeric_predictors())


svmlin_wf2 <- workflow(preprocessor = rec2, spec = svmlin_spec2)

svmlin_fit2 <- fit(svmlin_wf2, data = penguins.train)


## Predict and evaluate

penguins.test$pred_lin <- predict(
  svmlin_fit2,
  new_data = penguins.test,
  type = "raw"
)

# 3 classes-confusion matrix:
penguins.test |> conf_mat(species, pred_lin)
penguins.test |> accuracy(species, pred_lin)
# penguins.test |> sens(species, pred_lin, estimator = "macro_weighted")

pred_grid <- expand.grid(
  bill_length_mm = seq(30, 60, len = 101),
  body_mass_g = seq(2500, 6100, len = 101)
) |>
  augment(x = svmlin_fit2)

# Is this surprising? Not really, considering our predictors..
ggplot(pred_grid, aes(bill_length_mm, body_mass_g)) +
  geom_raster(aes(alpha = .pred_Chinstrap, fill = "Chinstrap")) +
  geom_raster(aes(alpha = .pred_Adelie, fill = "Adelie")) +
  geom_raster(aes(alpha = .pred_Gentoo, fill = .pred_class)) +
  geom_point(
    aes(fill = species),
    data = penguins.test,
    color = "black",
    shape = 21,
    size = 3,
    show.legend = FALSE
  ) +
  scale_alpha_continuous(
    name = expression(P(Species[c])),
    limits = c(0, 1),
    range = c(0, 1)
  ) +
  labs(
    fill = "Species"
  )


# SVR --------------------------------------------------------------------

# We can also perform support vector regression, if
# the response vector that is numerical rather than a factor.

svrlin_spec <- svm_linear(
  mode = "regression",
  engine = "kernlab",
  cost = tune(),
  margin = tune()
)
# We've added a new parameter - margin. The margin in SVR refers to the region
# around the regression line where errors are not penalized.

rec3 <- recipe(body_mass_g ~ bill_length_mm + species, data = penguins.train) |>
  step_naomit(body_mass_g) |>
  step_impute_mean(bill_length_mm) |>
  step_dummy(species, one_hot = TRUE) |>
  step_interact(~ starts_with("species"):bill_length_mm) |>
  step_normalize(all_numeric_predictors())

svrlin_wf <- workflow(preprocessor = rec3, spec = svrlin_spec)


# Tune the model...
folds <- vfold_cv(penguins.train, v = 5)

svrlin_grid <- grid_regular(
  cost(),
  margin = svm_margin(c(0, 2)),
  levels = c(5, 5)
)

svrlin_tune <- tune_grid(svrlin_wf, resamples = folds, grid = svrlin_grid)

autoplot(svrlin_tune)

# Finalize the model
svrlin_fit <-
  svrlin_wf |>
  finalize_workflow(select_best(svrlin_tune, metric = "rmse")) |>
  fit(data = penguins.train)


penguins.test$.pred <- as.vector(predict(
  svrlin_fit,
  new_data = penguins.test,
  type = "raw"
))
penguins.test |>
  group_by(species) |>
  rsq(body_mass_g, .pred)

ggplot(penguins.test, aes(bill_length_mm, body_mass_g, color = species)) +
  geom_point() +
  stat_smooth(aes(linetype = "OLS"), method = "lm", se = FALSE, geom = "line") +
  geom_line(aes(y = .pred, linetype = "SVR")) +
  labs(
    linetype = "model"
  )

# Exercise ---------------------------------------------------------------

# 1. Take one of the datasets used here (OJ / penguins) and predict the outcome
#   with 4 variables of your choice.
# 2. Use two method (linear, poly, radial) and compare their performance using
#   CV model comparison (see "03 Resampling and Tuning/02 model selection.R").
