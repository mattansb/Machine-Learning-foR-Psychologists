
library(rsample)
library(recipes)
library(yardstick)
library(caret)


# The data ----------------------------------------------------------------

data("Hitters", package = "ISLR")
Hitters <- na.omit(Hitters)

set.seed(1)
splits <- initial_split(Hitters, prop = 0.7)
Hitters_train <- training(splits)
Hitters_test <- testing(splits)


# PCR and PLS -------------------------------------------------------------

## (A) Prepossessing

rec <- recipe(Salary ~ ., data = Hitters_train) |> 
  # Standardize numeric variables:
  step_center(all_numeric_predictors()) |> 
  step_scale(all_numeric_predictors()) |> 
  # Make one hot encoding dummy variables
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

## PCR -----------------------------------
# This is a method that use PCA as a first step for predicting y. 
# IMPORTANT- this is an UNSUPERVISED method for dimension reduction - since PCA
# don't uses a response variable when building the components. There are other
# methods, which uses the response (hence, SUPERVISED), for using a similar
# goal- predicting y from components


## (B) Tune:

tg <- expand.grid(ncomp = 1:15)

tc <- trainControl(method = "cv", number = 10)

set.seed(44)
PCR_fit <- train(
  x = rec, 
  data = Hitters_train,
  method = "pcr",
  tuneGrid = tg,
  trControl = tc
)

PCR_fit$bestTune

summary(PCR_fit$finalModel)

# Note these are back transformed to the X predictor space.
coef(PCR_fit$finalModel, ncomp = 7)


## PLS ---------------------------
# We will now address the same problem using another dimension reduction method-
# PLS. Importantly, PCR is an UNSUPERVISED method and PLS is a SUPERVISED
# method. PLS counts for *Partial Least Squares*.

## (B) Tune:

PLS_fit <- train(
  x = rec,
  data = Hitters_train,
  method = "pls", # change method
  tuneGrid = tg, # same tune-grid
  trControl = tc
)

PLS_fit$bestTune

summary(PLS_fit$finalModel)

coef(PLS_fit$finalModel, ncomp = 3)


## Compare ----------------

cbind(coef(PCR_fit$finalModel, ncomp = 2),
      coef(PLS_fit$finalModel, ncomp = 2))


Hitters_test$PCR_pred <- predict(PCR_fit, newdata = Hitters_test)
Hitters_test$PLS_pred <- predict(PLS_fit, newdata = Hitters_test)

rmse(Hitters_test, Salary, PCR_pred)
rmse(Hitters_test, Salary, PLS_pred)

rsq(Hitters_test, Salary, PCR_pred)
rsq(Hitters_test, Salary, PLS_pred)
# In this case the performance is nearly identical


# Using PCA in other methods w/ {recipe} ----------------------------------

# What if I want to use KNN??
# We can still use PCA as part of our recipe (however, the number of dimensions
# is no longer a tunable hyperparameter). This can be achieved with
# `step_pca()`. There are two arguments that can be used to control how many PCs
# to save:
# - num_comp: the number of components
# - threshold: what proportion of variance should be saved?
# * Note that the predictors should be all be re-scaled _prior_ to the PCA step.


rec
# Already has a scaling step and centering step

rec_with_PCA <- rec |> 
  step_pca(all_numeric_predictors(), 
           threshold = 0.9) # give k PCs that represent 90% of the total variance

## (B) Tune:

tg <- expand.grid(k = c(1, 2, 5, 10, 20, 50, 100))

set.seed(44)
KNN_fit <- train(
  x = rec_with_PCA, 
  data = Hitters_train,
  method = "knn",
  tuneGrid = tg,
  trControl = tc
)

PLS_fit$recipe |> bake(new_data = NULL) # We had 22 predictors
KNN_fit$recipe |> bake(new_data = NULL) # We used 7 PCs

## Compare -------------------

Hitters_test$KNN_pred <- predict(KNN_fit, newdata = Hitters_test)

rsq(Hitters_test, Salary, PCR_pred)
rsq(Hitters_test, Salary, PLS_pred)
rsq(Hitters_test, Salary, KNN_pred)
# KNN with PCA is better...



# How about KNN without PCA?




