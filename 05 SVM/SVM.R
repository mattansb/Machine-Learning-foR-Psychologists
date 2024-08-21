

library(tidymodels)
# library(kernlab)


# The OJ Data -----------------------------------------------------------------

data("OJ", package = "ISLR")
?ISLR::OJ
# Which brand of orange juice was purchased?

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
rec <- recipe(Purchase ~ PriceDiff + LoyalCH, 
              data = OJ.train) |> 
  step_normalize(all_numeric_predictors())
# The decision boundary maximizes the distance to the nearest data points from
# different classes. Hence, the distance between data points affects the
# decision boundary SVM chooses. In other words, training an SVM over the scaled
# and non-scaled data leads to the generation of different models.



folds <- vfold_cv(OJ.train, v = 5)

oj_metrics <- metric_set(accuracy, sens, spec, f_meas)




# Support Vector Classifier --------------------------------------------------
# FITTING the support vector classifier


## Tune ------------------------

# We will fit the support vector classifier for a given
# value of the cost parameter. Here, we directly try more then one value and
# find the best fit with CV!
svmlin_spec <- svm_linear("classification", engine = "kernlab", 
                          cost = tune())

translate(svmlin_spec)
# Note that this engine converts the SVM to a probabilistic classifier
# using Platt scaling (a logistic regression model is fit to the SVM output).



svmlin_wf <- workflow(preprocessor = rec, spec = svmlin_spec)

svmlin_grid <- grid_regular(cost(), levels = 20)
# A cost argument allows us to specify the cost of a violation to the margin:
# small cost -> wide margins and many support vectors violate the margin.
# large cost -> narrow margins and few support vectors violate the margin.

svmlin_tune <- tune_grid(svmlin_wf, 
                         resamples = folds, 
                         grid = svmlin_grid, 
                         metrics = oj_metrics)

autoplot(svmlin_tune)

collect_metrics(svmlin_tune, type = "wide")

(svmlin_const <- select_by_one_std_err(svmlin_tune, desc(cost), metric = "f_meas"))

## The final model ------------------------------------------------------

svmlin_fit <- fit(finalize_workflow(svmlin_wf, svmlin_const), 
                  data = OJ.train)



# We can explore the model but extracting the underlying model object:
(svmlin_eng <- extract_fit_engine(svmlin_fit))
# training error is 0.170561 (for the cost 32)
# Number of Support Vectors : 358! 


# Here we demonstrated the use of this function on a two-dimensional example so
# that we can plot the resulting decision boundary.
X_train <- bake(extract_recipe(svmlin_fit), new_data = OJ.train, 
                all_predictors())

plot(svmlin_eng, data = X_train)
# Full circles\triangles = support vectors (365 obs.)
# Hollow circles\triangles = the remaining observations 




## Evaluating the support vector classifier performance ---------------------
# (on a new TEST DATA)

# Predict the class labels of these test observations. 
OJ.test$pred_lin <- predict(svmlin_fit, new_data = OJ.test, type = "raw")
OJ.test |> conf_mat(Purchase, pred_lin)
OJ.test |> oj_metrics(Purchase, estimate = pred_lin)








# Support Vector Machine ----------------------------------------
# (non-linear kernel)


## Polynomial ------------------------------------

svmpoly_spec <- svm_poly("classification", engine = "kernlab", 
                         cost = tune(), degree = 3, scale_factor = 1)
# Here we also add the degree argument to specify a degree for the polynomial
# kernel (e.g., for quadratic: degree = 2). Assuming the predictors have been
# standardized (which they should be), "scale_factor" can be set to 1.

translate(svmpoly_spec)

svmpoly_wf <- workflow(preprocessor = rec, spec = svmpoly_spec)


svmpoly_tune <- tune_grid(svmpoly_wf, 
                          resamples = folds, 
                          grid = svmlin_grid, # reuse cost grid
                          metrics = oj_metrics)

(svmpoly_params <- select_by_one_std_err(svmpoly_tune, desc(cost), metric = "f_meas"))



svmpoly_fit <- fit(finalize_workflow(svmpoly_wf, svmpoly_params), 
                   data = OJ.train)



(svmpoly_eng <- extract_fit_engine(svmpoly_fit))
# best training error is 0.179907 (for the cost 0.005)
# Number of Support Vectors : 470! 
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


svmrad_spec <- svm_rbf("classification", engine = "kernlab", 
                       cost = 32, # we get it...
                       rbf_sigma = 2)
# rbf_sigma is the sigma parameter for the radial basis kernel, which controls
# the smoothness of the decision boundary.

translate(svmrad_spec)

svmrad_wf <- workflow(preprocessor = rec, spec = svmrad_spec)

svmrad_fit <- fit(svmrad_wf, data = OJ.train)



(svmrad_eng <- extract_fit_engine(svmrad_fit))
# best training error is 0.172897 (for the cost 0.026)
# Number of Support Vectors : 588! 
plot(svmrad_eng, data = X_train)
# Yuck...


## Compare models ----------------------------------------------------------


OJ.test$pred_poly <- predict(svmpoly_fit, new_data = OJ.test, type = "raw")
OJ.test$pred_rad <- predict(svmrad_fit, new_data = OJ.test, type = "raw")


OJ.test |> oj_metrics(Purchase, estimate = pred_lin)
OJ.test |> oj_metrics(Purchase, estimate = pred_poly)
OJ.test |> oj_metrics(Purchase, estimate = pred_rad)
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


svmlin_spec2 <- svm_linear("classification", engine = "kernlab", 
                           cost = 0.1)

rec2 <- recipe(species ~ bill_length_mm + body_mass_g,
               data = penguins.train) |> 
  step_normalize(all_numeric_predictors())


svmlin_wf2 <- workflow(preprocessor = rec2, spec = svmlin_spec2)

svmlin_fit2 <- fit(svmlin_wf2, data = penguins.train)



## Predict and evaluate

penguins.test$pred_lin <- predict(svmlin_fit2, new_data = penguins.test, type = "raw")

# 3 classes-confusion matrix:
penguins.test |> conf_mat(species, pred_lin)
penguins.test |> accuracy(species, pred_lin)
# penguins.test |> sens(species, pred_lin, estimator = "macro_weighted")

# Is this surprising? Not really, considering our predictors..
ggplot(penguins.test, aes(bill_length_mm, body_mass_g)) +
  geom_point(aes(color = species))


# SVR --------------------------------------------------------------------

# We can also perform support vector regression, if
# the response vector that is numerical rather than a factor.

svrlin_spec <- svm_linear(mode = "regression", engine = "kernlab", 
                          cost = tune(), margin = tune())
# We've added a new parameter - margin. The margin in SVR refers to the region
# around the regression line where errors are not penalized.

rec3 <- recipe(body_mass_g ~ bill_length_mm + species, 
               data = penguins.train) |> 
  step_naomit(body_mass_g) |> 
  step_impute_mean(bill_length_mm) |> 
  step_dummy(species, one_hot = TRUE) |> 
  step_interact(~starts_with("species"):bill_length_mm) |> 
  step_normalize(all_numeric_predictors())
  
svrlin_wf <- workflow(preprocessor = rec3, spec = svrlin_spec)




# Tune the model...
folds <- vfold_cv(penguins.train, v = 5)

svrlin_grid <- grid_regular(cost(), margin = svm_margin(c(0, 2)), 
                            levels = c(5, 5))

svrlin_tune <- tune_grid(svrlin_wf, resamples = folds, grid = svrlin_grid)

autoplot(svrlin_tune)

# Finilize the model
svrlin_fit <- 
  svrlin_wf |> 
  finalize_workflow(select_best(svrlin_tune, metric = "rmse")) |> 
  fit(data = penguins.train)




penguins.test$.pred <- as.vector(predict(svrlin_fit, new_data = penguins.test, type = "raw"))
penguins.test |> 
  group_by(species) |> 
  rsq(body_mass_g, .pred)

ggplot(penguins.test, aes(bill_length_mm, body_mass_g, color = species)) + 
  geom_point() + 
  geom_line(aes(y = .pred))


# Exercise ---------------------------------------------------------------

# 1. Take one of the datasets used here (OJ / penguins) and predict the outcome
#   with 4 variables of your choice.
# 2. Use two method (linear, poly, radial) and compare their performance using
#   CV model comparison.



