# All tutorial materials were originally developed by Yael Bar-Shachar (no relation).



                                ### ML- TUTORIAL 1- Intro ###

## CARET package---------------------------------------------------------------

# We will also use "caret": 
library(caret)

# The caret package (short for Classification And REgression Training) is a set
# of functions that attempt to streamline the process for creating predictive
# models. The package contains tools for:

# - model tuning using re-sampling
# - model fitting and prediction

# (see: https://topepo.github.io/caret/)

# THE DATA- Auto & THE PROBLEM\ QUESTION- regression : ------------------------

# For datasets we will mostly use the Book official package ISLR
library(ISLR) # The "ISLR" package includes data sets we will use this semester
data("Auto")

# The Auto Dataset contains information about ... cars.
# For each car, the following vars were recorded:
#  - cylinders
#     Number of cylinders between 4 and 8
# - displacement
#     Engine displacement (cu. inches)
# - horsepower
#     Engine horsepower
# - weight
#     Vehicle weight (lbs.)
# - acceleration
#     Time to accelerate from 0 to 60 mph (sec.)
# - year
#     Model year (modulo 100)
# - origin
#     Origin of car (1. American, 2. European, 3. Japanese)
Auto$origin <- factor(Auto$origin)
# - name
#     Vehicle name
?Auto
# What we are interested is gas consumption: MPG (miles per gallon)

# Getting to know the data:
dim(Auto)
names(Auto)
head(Auto)
str(Auto)

# Initial Data Splitting - Train & Test Data ----------------------------------

library(rsample) # part of tidymodels (data splitting and resampling)


# We will TRAIN the model (i.e. fit) on the 70% of the observations randomly
# assigned and TEST the model (i.e. predict and assess performance) on the 30%
# that were left.

# We can use the {rsample} package. More examples of data splitting:
# https://rsample.tidymodels.org/

set.seed(1) # because we will use random sampling we need to set a 
# random seed in order to replicate the results

splits <- initial_split(Auto, prop = 0.7)
train.data <- training(splits) # We will train the model using 70% of the obs.
test.data <- testing(splits) # We will test the model on 30% of the obs.


# The general process for all types of questions (regression\ classification)
# and fitting methods will be:
#
# (A) Prepossessing >>> 
#     (B) Fitting >>> 
#         (C) Predicting >>> 
#             (D) Assessing performance


## (A) Prepossessing ---------------------------------------------------------
# (on the TRAIN data)

# There several functions to pre-process the predictor data (i.e. prepare the
# data for fitting process). One very common function is for centering and
# scaling the predictors.
# For knn this is VERY IMPORTANT - Because KNN method identify the observations
# according to their **distance**, the scale of the variables matters: large
# scale -> larger distance between the observations on that X.

# WHY DO WE DO THIS ON THE TRAINING DATA?

library(recipes) # data pre-processing pipeline

rec <- recipe(mpg ~ horsepower + weight, # model specification
              data = train.data) # the data
rec

# There are many prepossessing "steps" we can take:
# https://recipes.tidymodels.org/reference/index.html

rec <- rec |> 
  step_center(all_predictors()) |> 
  step_scale(all_predictors())
rec

prep(rec)

bake(prep(rec), new_data = NULL) |> head()

bake(prep(rec), new_data = test.data) |> head() # what is going on here?

## (B) MODEL FITTING / LEARNING / TRAINING using train() -----------------------
# (on the TRAIN data)

# For the fitting process we will use the train() function from caret which
# train a model using a variety of functions (this time- KNN)

# But one small thing before....
set.seed(1)
# We set a random seed because there is a resampling process which is inherent
# to KNN method - the process of searching for the neighbors. If several
# observations are tied as nearest neighbors, R will randomly break the tie.
# (you don't have to understand this part). Therefore, a seed must be set in
# order to ensure reproducibility of results.


# lets' explore train():

tg <- expand.grid(
  k = 5 # [1, N] neighbors 
)
# (An OPTIONAL\MUST argument depends on the method used)
# A data frame with possible tuning values (also called hyperparameters).
# For knn we MUST enter the k. (Or do we?)
# For now we will use only one k for the fitting procedure. (next time we will
# see that we will be able to test for different hyperparameters).

tc <- trainControl(method = "none")
# (an OPTIONAL argument - option which we usually use for ML)
# method of controlling the computational process of fitting.
# For now, we used "none" i.e., now this argument does nothing.
# Next time it will mean a lot!

knn.fit5 <- train(
  x = rec, # the recipe does two things: 
  # (1) informs the model what the y and what the Xs are 
  # (2) what pre-proc steps should be taken.
  data = train.data, # the training data (a MUST argument)
  method = "knn", # method used for fitting the model (now - knn)
  tuneGrid = tg,
  trControl = tc
)

# Here are all available fitting methods within train(): 
# https://topepo.github.io/caret/available-models.html
# Note - for each model you can check for each TYPE of question it is used and
# what are the tuning parameters and/or hyperparameters.




## (C) PREDICTING using predict() ---------------------------------------------
# (using the fitted model, for prediction on the TRAIN or TEST data)


# predict() function uses the fitted model + given values of the predictors to
# predict the values Y.

test.data$mpg_hat <- predict(knn.fit5,  # fitted model used for prediction
                             newdata = test.data)  # the data to predict from
# Values for the PREDICTORS will be taken from the TEST data. Note that
# predict() *also* processes the newdata according to the trained recipe!

plot(mpg ~ mpg_hat, data = test.data) # true vs predicted values



## (D) Assessing model performance on the TEST data ---------------------------------------------------------------------

# How we assess model performance? 
# For regression problems- R-squared, MSE, RMSE, MAE...

c(
  Rsq = cor(test.data$mpg, test.data$mpg_hat)^2,
  RMSE = sqrt(mean((test.data$mpg - test.data$mpg_hat)^2)),
  MAE = mean(abs(test.data$mpg - test.data$mpg_hat))
)

# or:

library(yardstick)

c(
  Rsq = rsq_vec(truth = test.data$mpg, estimate = test.data$mpg_hat),
  RMSE = rmse_vec(truth = test.data$mpg, estimate = test.data$mpg_hat),
  MAE = mae_vec(truth = test.data$mpg, estimate = test.data$mpg_hat)
)

# # Or you can do this.. But you won't really need to...
# quant_metrics <- metric_set(rsq, rmse, mae)
# quant_metrics(test.data, truth = mpg, estimate = mpg_hat)



## Playing with K --------------------------------------------------------------------------------------------------

# We chose K=5, but what will happen if we will use bigger k? like 10?

# (B) Fitting the same model using KNN (with k=10) on the train data:
tg2 <- expand.grid(
  k = 10 # [1, N] neighbors 
)

knn.fit10 <- train(
  x = rec,
  data = train.data,
  method = "knn",
  tuneGrid = tg2,
  trControl = tc
)

# (C) PREDICTING for the test data
test.data$mpg_hat_2 <- predict(knn.fit10, newdata = test.data)

# (D) ASSESSING performance
c(
  Rsq = rsq_vec(truth = test.data$mpg, estimate = test.data$mpg_hat_2),
  RMSE = rmse_vec(truth = test.data$mpg, estimate = test.data$mpg_hat_2),
  MAE = mae_vec(truth = test.data$mpg, estimate = test.data$mpg_hat_2)
)
# Gives very similar results to K=5...



## Playing with the model itself -----------------------------------------------------------------

# Well, maybe the problem is that 
# mpg ~ horsepower + weight,
# is just not a good model....

# That make sense as I know nothing about cars...
# (hopefully for our research questions we use more thinking)

# Let's try to use all the 8 available predictors in the data.
# We can list all of them, or just use: Direction ~ . 

# (A) pre-proc

rec2 <- recipe(mpg ~ ., # all predictors,
               data = train.data) |>
  step_rm(name) |> # EXCEPT for the name of the car-model (meaningless!)
  step_center(all_numeric()) |> # Note the use of all_numeric
  step_scale(all_numeric()) |> 
  step_dummy(all_factor(), # Make dummy variables (we'll talk about this later!)
             one_hot = TRUE)

bake(prep(rec2), new_data = NULL) |> head()
# Note that the order matters - where we put step_dummy() determines if the
# dummies will be centered and scaled!


# (B) Fitting the model:
knn.fit10.B <- train(
  x = rec2,
  data = train.data,
  method = "knn",
  tuneGrid = tg2,
  trControl = tc
)

# (C) PREDICTING for the test data
test.data$mpg_hat_3 <- predict(knn.fit10.B, newdata = test.data)


# (D) ASSESSING performance

c(
  Rsq = rsq_vec(truth = test.data$mpg, estimate = test.data$mpg_hat_3),
  RMSE = rmse_vec(truth = test.data$mpg, estimate = test.data$mpg_hat_3),
  MAE = mae_vec(truth = test.data$mpg, estimate = test.data$mpg_hat_3)
)
plot(mpg ~ mpg_hat_3, data = test.data)

# Rsq is better! But what happened to RMSE/MAE?? And the plot??
# Go back and fix it!


# Exercise ---------------------------------------------------------------------

## New Data!
# Wage dataset from ISLR includes 3000 obs with 10 predictors 
?Wage

# We want to predict `wage`.


## Your task:

# Note- when needed- use set.seed(1) for replicability

# (1) Split the data to train and test (use p=0.7)
# (2) Predict wage out 3 of the other variables with a knn model with 
#     (k=5). That is, fit and predict.
# (3) Assess performance using the metrics you've learned
# (4) To improve flexibility, try a different k. Will you use bigger\ smaller k?
# (5) If you tried smaller k try now bigger k (or vice versa). what will you
#     earn from this and what will you lose? (in terms of performance indices)


