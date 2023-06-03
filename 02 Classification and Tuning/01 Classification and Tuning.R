                      ### ML - TUTORIAL 2  ###

# Load packages:
library(dplyr)
library(ISLR)

library(caret) 
library(rsample)
library(yardstick)
library(recipes)


# Warming up- a classification problem with logistic regression ---------------

# In Tutorial 1 we've used caret for a regression problem. Today we are looking
# at classification with the Smarket dataset:
# Smarket (from ISLR) consists of percentage returns for the a stock index over
# 1,250 days. For each date, the following vars were recorded:
# - Lag1 through Lag5 - percentage returns for each of the five previous trading
#   days.
# - Volume - the number of shares traded on the previous day(in billions).
# - Today - the percentage return on the date in question.
# - Direction - whether the market was Up or Down on this date.

# Assume the following classification task on the Smarket data:
# predict Direction (Up/Down) using the features Lag1 and Lag2.
# If we are not sure how Direction is coded we can use contrasts():
contrasts(Smarket$Direction)

table(Smarket$Direction)
# The base rate probability:
648 / (648 + 602)
proportions(table(Smarket$Direction))

# We'll start by using a parametric method - logistic regression:

# Data Splitting (SAME as in tutorial 1):
set.seed(1234)
splits <- initial_split(Smarket, prop = 0.7)
Smarket.train <- training(splits)
Smarket.test <- testing(splits)

## (A) Fitting logistic regression on train data using caret --------------
tc <- trainControl(method = "none") # remember "none" = no training method.
                                    # for the warmup we will leave it like that...
                                  
# We will use train().
LogRegfit <- train(
  Direction ~ Lag1 + Lag2, # model syntax
  data = Smarket.train, # the data
  method = "glm", family = binomial("logit"), # For logistic regression
  trControl = tc # although this is redundant when set to "none"
)

# NOTE 1 - here we don't have a tuneGrid with possible hyperparameters as these
# are not needed for logistic regression. We also don't need to preprocess that
# data, since our simple straightforward logistic regression don't require any
# pre processing. (We can if we wanted to, though.)

# NOTE 2 - yes, not using any training method would leave us with the same
# results as if we've used a simple glm() function with "binomal" argument. This
# is how we would do this using regular R method:
# LogRegfit <- glm(Direction ~ Lag1 + Lag2,
#                  data = Smarket.train,
#                  family = binomial("logit"))
# But... We want the power of {caret}!



# You _can_ interpret the model coefficients using exp just as you've learned in
# other courses... But, let's NOT, and instead focus on the process in the ML
# course...

## (B) Prediction using predict() ------------------------------------------
predicted.probs <- predict(LogRegfit, newdata = Smarket.test, type = "prob")
# output probabilities of the form:
# p(Y = 1|X) -> p(direction is UP given the specific Xs).
# This is relevant for classification problems of course
head(predicted.probs)  # predicted outcomes for the first 6 obs.
predicted.probs[1:10,"Up"]

# Here we predicted probabilities, but what if we want to predict classes?
# use "raw" instead of "prob" and get the class prediction (based on 0.5 cutoff)
predicted.classes <- predict(LogRegfit, newdata = Smarket.test, type = "raw") 
# raw is the default so this will give us the same:
predicted.classes <- predict(LogRegfit, newdata = Smarket.test) 

predicted.classes[1:10] # predicted.classes for the market to go up for the 10 first observations

# OR, if from some reason we don't want to use the 0.5 cutoff, we can convert
# the predicted probabilities to a binary variable based on selected cutoff
# criteria. E.g. for making a binary decision of > 0.7
predicted.classes2 <- factor(predicted.probs[["Up"]] > 0.9,
                             levels = c(FALSE, TRUE),
                             labels = c("Down", "Up"))
# as columns complete to 1.
predicted.classes2[1:10]


## (C) Assessing model performance -------------------------------------------

# How we assess model performance? 
# For regression problems- MSE, RMSE...
# For classification problems- performance indices based on the confusion matrix
confusionMatrix(predicted.classes,       # The predicted classes 
                Smarket.test$Direction,  # The Reference classes (Real classes) 
                positive = "Down")
# Note - 'Positive' Class : Down
# That is, "Hit"\"TP" will be- saying the market is Down when it is really Down
# an "False alarm"\"FP" will be- saying the market is Down when it is really Up

# It seems that it will be more intuitive to look at "Up" as positive, we can
# flip it using "positive" argument:
confusionMatrix(predicted.classes,
                Smarket.test$Direction,
                positive = "Up") 

# For example,  
# One performance indices is Accuracy (ACC) - 
# ratio of correct predictions (fraction of days for which the prediction was correct):
(40 + 161)/374
# That is- TEST ERROR is 1-0.4626=0.5374

# Also- sensitivity, recall, hit rate, or true positive rate (TPR) are all the
# same names for:
161/(33 + 161)
# https://en.wikipedia.org/wiki/Confusion_matrix for more about terminology and
# derivations of a confusion matrix

# ANYWAY, all indices tells us that this model wasn't that amazing (for accuracy
# flipping a coin would be better...)

## We can also look at the ROC curves!

# for creating ROC curve we need to create a cloumn of the predicted probs for our "positive"
# class ("Up") within the test data:
Smarket.test$prob_logisticReg <- predicted.probs[["Up"]]
head(Smarket.test)

# we will use yardstick::roc_curve():
Smarket.test |> 
  roc_curve(Direction, prob_logisticReg, 
          event_level = "second") |> 
  ggplot2::autoplot()

# Here we can see how our modeled classifier acts (in terms of True Positive
# Rate and False Positive Rates) using different thresholds. It seems that for
# some varied thresholds our classifier isn't much better than a random
# classifier.


# Cross-Validation exemplified on KNN model -----------------------------------

# We will show how Cross-Validation can help us choose the hyperparamater k for
# KNN.

# Let's take another CLASSIFICATION problem on this data. but with few changes:
# 1. change the model itself- predict Direction from all predictors!
# 2. change the method- KNN (with binary outcome) instead of logistic regression
# 3. TUNE THE MODEL WHILE FITTING. That is - use CV. Specifically, use CV to
#   also choose the best K.


## Leave-One-Out Cross-Validation- LOOCV --------------------------------------

# One type of CV is LOOCV.

# For N obs. we will re-sample the data n times and for each sample: n-1 obs.
# will be the training set, and the one left out obs. will be the validation
# set.

# How?
# Finally using "trainControl"!
# The LOOCV estimate can be automatically computed for any fitted model with
# train(). The cool thing about caret is that it enables us to use the
# "trainControl" argument when fitting any model, so we don't need to use any
# other packages that are specific for preforming CV to each type of models
# (e.g. boot).

## (A) FITTING a KNN model using LOOCV 

rec <- recipe(Direction ~ .,
              data = Smarket.train) |> 
  step_range(all_numeric_predictors()) # re-range to 0-1


tc <- trainControl(method = "LOOCV", 
                   selectionFunction = "best") 
# remember we used "none" in the previous examples? that ment we told R 'don't
# use resampling when fitting the data' now we tell R to do resampling, and
# specifically- LOOCV We can just use this fitting method to make the fitting
# proceadure to be more reliable (for any chosen k), or, we can also make use of
# it in order to CHOOSE the best K (i.e., hyperparameter)!

# Let's try now some options for k to better understand:
tg <- expand.grid(k = c(5, 10, 50, 200))

set.seed(12345)  # For KNN fitting process (see tutorial 1)

knn.fit.LOOCV <- train(
  x = rec, 
  data = Smarket.train, 
  method = "knn", # knn instead of logistic regression
  tuneGrid = tg, # our K's for KNN
  trControl = tc, # the method for tuning (LOOCV)
  metric = "Accuracy"
)
# NOTE 1: this specify the summary metric will be used to select the optimal
# model. By default, possible values are "RMSE" and "Rsquared" for regression
# and "Accuracy" and "Kappa" for classification. We can specify custom metrics.

# NOTE 2: caret uses a threshold of 0.5 by default!


# (notice the time it takes to fit using LOOCV!)

knn.fit.LOOCV$results 
# Here we see accuracy for all values of K
# 1- accuracy will be the VALIDATION ERROR
# (it is somewhat in-between train and test errors)
# *Kappa is a metric that compares an Observed Accuracy with an 
# Expected Accuracy (random chance).  

val.error <- 1 - knn.fit.LOOCV$results$Accuracy
plot(knn.fit.LOOCV$results$k, val.error)
knn.fit.LOOCV$bestTune 

# Best K is k=50 , where Accuracy = 0.899 and the validation error is
# 1 - 0.899 = 0.101

# Final model is automatically chosen based on the best tuning hyperparameter(s)
# (for now only one hyperparameter - k) is set to k=50
knn.fit.LOOCV$finalModel

## (B) PREDICTING on test data:
predicted.classes.LOOCV <- predict(knn.fit.LOOCV, newdata = Smarket.test, type = "raw") 

## (C) Assessing performance: 
confusionMatrix(predicted.classes.LOOCV, Smarket.test$Direction, positive = "Up")
# Accuracy (0.91) is great, even in contrast to the validation error! 
# It might be that our model is just good\or that CV help us to create a stable
# model (also, remember that accuracy isn't everything and we have many
# other fit indices that deserve attention) 

# roc curve:
Smarket.test$prob_KNN5 <- predict(knn.fit.LOOCV, newdata = Smarket.test, type = "prob")[, "Up"]
Smarket.test |> 
  roc_curve(Direction, prob_KNN5, 
            event_level = "second") |> 
  ggplot2::autoplot()
# pretty!!!


## k-Fold Cross-Validation -----------------------------------------------------------------

# LOOCV might be time consuming, and K-fold CV can give us very similar results...

# In K-folds CV we split data into k parts and re-sample these parts k times.
# For each round of sampling k-1 parts are used to train the model, 
# and one left out part of the data remains as the validation set.
# let's try k = 10 (a common choice for k).

## (A) FITTING a KNN model using 10-folds CV:

tc <- trainControl(method = "cv", number = 10,
                   selectionFunction = "best")
# for preforming k-Fold Cross-Validation just switch "trainControl" to:
# trainControl(method = "cv", number = k, selectionFunction = "best")

# Let's try again the same options for knn's k:
tg

set.seed(9) 
# For KNN fitting process, but also- we must set a random seed for, since the
# obs. are sampled into the one of the k folds randomly

knn.fit.10CV <- train(
  x = rec, 
  data = Smarket.train, 
  method = "knn",
  tuneGrid = tg,
  trControl = tc,
  metric = "Accuracy"
)

knn.fit.10CV$bestTune # The best tuning parameter based on 10-folds cv is 50
knn.fit.10CV$results # we can see mean accuracy (across all folds for a given
                     # k neighbors, as well as Accuracy SD)
# Average validation error?


## (B) PREDICTING on test data:
predicted.classes.10CV <-
  predict(knn.fit.10CV, newdata = Smarket.test, type = "raw") 

## (C) Assessing performance: 
confusionMatrix(predicted.classes.10CV, Smarket.test$Direction, positive = "Up")
# ACC on test set based on the 50-nearest neighbor model that was fitted using
# the 10-fold CV is: 0.91
# Test error: 1-0.91= 0.9
# which is at least as good as the LOOCV fit...


# IMPORTANT NOTE!
# THE BEST is to use LOOCV\ K-folds CV when fitting the train data, 
# and to test the fitted model with new test data.

# BUT, for very small samples and\or for very low base rates we might
# choose to not split to train and test data sets so we would be able
# to utilize all available data. Still, using CV while fitting will 
# help us getting a more reliable model and to choose hyperparameters.



# Exercises: ----------------------------------------------------------------------

# Note- * when needed- use set.seed({some number}) for replicability
#       * work with split to train =70% of the data (and test= 30%)

# A) Use the Smarket dataset and predict Direction from Lag1 + Lag2 + Lag3.
#   But now fit a *logistic regression* using 10-folds CV and assess
#   performance on test data.
#   What are the cv error and test error?

# B) Fit knn while choosing the best K with 10-folds CV, use the Caravan
#   dataset:

# Caravan dataset includes 85 predictors that measure demographic
# characteristics for 5,822 individuals. The response variable (column 86) is:
# Purchase- whether or not a given individual purchases a caravan insurance
# policy.
Caravan$Purchase # Purchase is a factor with 2 levels "No","Yes"
str(Caravan) # all other variables are numeric
psych::describe(Caravan)
# In this task we will predict Purchase out this variables:
# MOSTYPE,MOSHOOFD, MOPLLAAG

# 1. Fit KNN with k=1, 5 and 20 using 10-folds CV and assess performance on test data.
#    what were the chosen tuning parameter, cv error and test error?
# 2. Fit KNN with k=1, 5 and 20 using LOOCV and assess performance on test data.
#    Fitting time will be much longer than the time it took to fit the knn model
#     on the Auto dataset. How can you explain it?
