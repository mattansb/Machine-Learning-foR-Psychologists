### Tutorial 3 - Model Selection and Regularization ###

library(dplyr)
library(ggplot2)

library(ISLR)

library(caret)
library(rsample)
library(yardstick)
library(recipes)

# Recommendation: read "Feature Selection Overview" on caret website:
# https://topepo.github.io/caret/feature-selection-overview.html


# Hitters DATA and the PROBLEM ---------------------------------
# (a REGRESSION problem)

# Hitters Dataset: Baseball Data from the 1986 and 1987 seasons
data("Hitters")

# A data frame with 322 observations of major league players on the following 20
# variables.
dim(Hitters)
names(Hitters)
?Hitters

# We wish to predict a baseball player's *Salary* on the basis of preformence
# variables in the previous year.
# Which 19 predictors will be best for predicting Salary?

# A little note regarding our dependent variable:
sum(is.na(Hitters$Salary)) # Salary variable is missing for 59 of the players.
Hitters <- na.omit(Hitters) # USUALLY THIS IS A BIG NO-NO
dim(Hitters)               #we are now left with 263 rows with full data


# Split:
set.seed(123442) 
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
Hitters.test <- testing(splits)



# PART A: Model Selection - Best Subset Selection Method ----------------------

# We will see how to fit Best Subset Selection.

# Our data is REALLY SMALL such that splitting the data to train and test might
# leave us with very small datasets. Let's focus with finding the best subset of
# features for the full data. When possible, we will want to first split the
# data, select features on the train data and test them on the test data.

# Best Subset Selection is generally better than the stepwise/forward/backward
# methods (see the lesson 3) However, if you really want to- see the code
# "Extra- Forward, Backward and Stepwise selection"

# With caret we can't use the basic sequential best subset selection...! ):

# Therefore, we will use the regsubsets() function from leaps package:
library(leaps)

# regsubsets() performs best subset Selection by identifying the best model that
# contains a given number of predictors, where (best is quantified using RSS).

regfit.full <- regsubsets(Salary ~ ., data = Hitters.train)

summary(regfit.full) 
# outputs the best set of variables for each model size up to the best
# 8-variable model (8 is the default).
# An asterisk indicates the variable is included in the corresponding model. 
# For instance, this output indicates that:
# the 2-variable model contains: Hits and CHits.  
# the 3-variable model contains: Walks, CatBat, and CHits.  

# If we want we can fit in this data up to a 19-variable model (and not 8) using
# the nvmax option.
regfit.full <- regsubsets(Salary ~ ., data = Hitters.train, nvmax = 19)
summary(regfit.full)

# Let's get a closer look on the statistics of this output:
reg.summary <- summary(regfit.full)
names(reg.summary) 
# The summary() function also returns R2 (rsq), RSS, adjusted R2 (adjr2), Cp,
# and BIC. We can examine these statistics to select the best overall model:

plot(reg.summary$rsq)
# For instance, we see that R2 increases from 32% for 1-variable model, to
# almost 55%, for 19-variables model. As expected, the R2 increases
# monotonically as more variables are included!

plot(reg.summary$adjr2) # This is not the case for adjusted R2!

# Plotting RSS,adj.R2,Cp and BIC for all of the models at once will help us
# decide which model to select:

reg.summary_df_long <- reg.summary[2:6] |> 
  as.data.frame() |> 
  mutate(nv = 1:n()) |> 
  tidyr::pivot_longer(cols = -nv, 
                      names_to = "Index", 
                      values_to = "value")

ggplot(reg.summary_df_long, aes(nv, value)) +
  facet_wrap(~Index, scales = "free") +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = 1:19, minor_breaks = NULL)

# regsubsets() has a built-in plot() command for displaying the selected
# variables for the best model with a given predictors num., ranked (top to
# bottom) according to the BIC, Cp, adjusted R2, or AIC:
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
# The top row of each plot contains a black square for each variable selected
# according to the optimal model associated with that statistic. E.G. several
# models share a BIC close to -150. However, the model with the lowest BIC (top
# row) is 6-variable model that contains: AtBat, Hits, Walks, CRBI, DivisionW,
# and PutOuts.

# BIC places the heaviest penalty on models with many variables.


## Two difficulties with regsubsets() ----------------------------------------------

# If data was good enough for train and test, we would have wanted to assess
# performance of best subset selection on test data.there is no predict() method
# for regsubsets()

### Problem 1 - no predict() for regsubsets()! ---------------
# So we need a HOME-MADE 'predict.regsubsets' FUNCTION: 
predict.regsubsets <- function(object, newdata, id, ...) {
  form <- as.formula(object$call[[2]])
  mm <- model.matrix(form, newdata)
  
  coefi <- coef(object, id = id)
  
  xvars <- names(coefi)
  as.vector(mm[, xvars] %*% coefi)
}

# Manually choose the best model by metric on the train set:
which.max(reg.summary$adjr2) # 9
which.min(reg.summary$cp)  # 9
which.min(reg.summary$bic) # 5

# different metrics suggest different things...
# it may be (also) related to the fact that we have very small data.

# (2) Let's PREDICT on test data when we take the best 14 predictors model
Hitters.test$pred9 <- predict(regfit.full, newdata = Hitters.test,
                              id = 9) # chosen variable number
# And 6:
Hitters.test$pred5 <- predict(regfit.full, newdata = Hitters.test, 
                              id = 5) 

# (3) Finally - assess performance:
my_metrics <- metric_set(rsq, rmse)
my_metrics(Hitters.test, truth = Salary, estimate = pred9)
my_metrics(Hitters.test, truth = Salary, estimate = pred5)

# Which is best?




### Problem 2 - If we want CV, we need to build a loop! ----------

# Let's run CV on the FULL data (because, again, data is very small, and using
# CV will make us more confident in the model fitted):
splits <- vfold_cv(Hitters, v = 10)

cv.errors <- matrix(NA, nrow = 10, ncol = 19)

for (i in seq_len(10)) {
  i_splits <- splits$splits[[i]]
  hold_out <- assessment(i_splits)
  training_set <- analysis(i_splits)
  
  best.fit <- regsubsets(Salary ~ ., data = training_set,
                         nvmax = 19)
  
  for (n in seq_len(19)) {
    # predict for each model size in test set
    pred <- predict(best.fit, newdata = hold_out, id = n)
    # using our home-made predict() function!
    
    
    cv.errors[i,n] <- rmse_vec(hold_out$Salary, pred)
  }
}

data.frame(
  n = 1:19,
  rmse = apply(cv.errors, 2, mean),
  rmseSD = apply(cv.errors, 2, sd)
)
# Which to choose?



# PART B - Regularization: Shrinkage Methods ----------------------

## Ridge Regression -------------------------

# We will perform ridge regression and the lasso in order to predict Salary on
# the Hitters data.

## (A) Prepossessing
rec <- recipe(Salary ~ ., data = Hitters.train) |> 
  step_dummy(all_factor_predictors()) |> 
  step_scale(all_numeric_predictors())
# IMPORTANT! scale the variable pre-fitting



## (B) Tune:
# One hyperparameter will be the type of penalty: *alpha* argument determines
# what type of model is fit based on the penalty- alpha=0 for a ridge regression
# model, alpha=1 for lasso model, and 0<alpha<1 for net...we will start with
# ridge.
# Another hyperparameter will be the tuning penalty lambda. Let's choose a range
# for lambda values.

tg <- expand.grid(
  alpha = 0, # [0, 1]
  lambda = 2 ^ seq(10, -2, length = 50) # [0, Inf]
)

# Here we will implement it over a grid of 100 values ranging from lambda=10^10
# to lambda=10^-2, thus covering the full range of scenarios from the null model
# containing only the intercept, to the least squares fit (lambda almost 0).

tc <- trainControl(method = "cv", number = 5)

# Fit ridge regression with 10-folds cv:

set.seed(1)
rigreg_fit <- train(
  x = rec,
  data = Hitters.train,
  method = "glmnet",
  tuneGrid = tg,
  trControl =  tc
)



rigreg_fit # we can see CV errors for each lambda- and the chosen best lambda
rigreg_fit$bestTune # gives the row number for best lambda

#lets see it:
plot(rigreg_fit, xTrans = log)
# x - log(lambda), if we won't use log() it will be very hard to see 
# y - RMSE 
bestlambda <- rigreg_fit$bestTune$lambda
log(bestlambda) # seems reasonable, looking at the plot.



# (C) Predict:
Hitters.test$ridge.pred <- predict(rigreg_fit, newdata = Hitters.test)
# Uses the best lambda

# EVALUATE the RMSE on the TEST set, associated with this value of lambda:
rmse(Hitters.test, truth = Salary, estimate = ridge.pred)
rsq(Hitters.test, truth = Salary, estimate = ridge.pred)


# (D) Coefficients:
## We can extract the model's coefficients according to lambda:
# The coef of the chosen model:
coef(rigreg_fit$finalModel, s = bestlambda)   
# or other lambdas...
# E.g. for lambda = 0.0000, (this result should be similar to OLS result)
coef(rigreg_fit$finalModel, s = 0)


# the different models produced by different lambdas! 
# and the parameters gets smaller as lambda rises:
#We can see that depending on the choice of tuning
#parameter, more coefficients will be exactly equal to zero:
plot(coef(rigreg_fit$finalModel, s = 0), ylab = "Value")
plot(coef(rigreg_fit$finalModel, s = 100), ylab = "Value")
plot(coef(rigreg_fit$finalModel, s = 1000), ylab = "Value")

# We can see that the Ridge penalty shrink all coefficients, but doesn't set any
# of them exactly to zero. Ridge regression does not perform variable selection!
#
# This may not be a problem for prediction accuracy, but it can create a
# challenge in model interpretation in settings in which the number of variables
# is large.

# The lasso method overcomes this disadvantage...


## The Lasso -------------------------------------

# As with ridge regression, the lasso shrinks the coefficient estimates towards
# zero. However, Lasso's penalty also force some of the coefficient estimates to
# be exactly equal to zero (when lambda is sufficiently large). Hence, performs
# variable selection.

# We once again use the train() function; however, this time we use the argument
# alpha=1. Other than that change, we proceed just as we did in fitting a ridge
# model.

# (B) TUNE:

tg <- expand.grid(
  alpha = 1, # [0, 1] switch to alpha=1 for lasso
  lambda = c(2 ^ seq(10,-2, length = 50)) # [0, Inf] SAME lambdas
) 

set.seed(1)
lasso_fit <- train(
  x = rec,
  data = Hitters.train,
  method = "glmnet",
  tuneGrid = tg,
  trControl =  tc
)
lasso_fit

#We can see that depending on the choice of tuning
#parameter, more coefficients will be EXACTLY equal to zero:
plot(coef(lasso_fit$finalModel, s = 0), ylab = "Value")
plot(coef(lasso_fit$finalModel, s = 50), ylab = "Value")
plot(coef(lasso_fit$finalModel, s = 100), ylab = "Value")
plot(coef(lasso_fit$finalModel, s = 1000), ylab = "Value")

# the best tuning parameter:
plot(lasso_fit, xTrans = log)
bestlambda <- lasso_fit$bestTune$lambda
log(bestlambda)
bestlambda

# (C) PREDICTING for test data (using best lambda).
Hitters.test$lasso.pred <- predict(lasso_fit, newdata = Hitters.test) 

rmse(Hitters.test, truth = Salary, estimate = lasso.pred)
rsq(Hitters.test, truth = Salary, estimate = lasso.pred)
# for ridge it was slightly smaller, but this one is much more parsimonious:
# 7 predictors instead of 19!
(coef <- coef(lasso_fit$finalModel, s = bestlambda))
sum(coef==0) # some coefs are exactly 0!
plot(coef)


## Elastic Net---------------------------------------------------------------

# Elastic Net emerged as a result of critique on lasso, whose variable selection
# can be too dependent on data and thus unstable. The solution is to combine the
# penalties of ridge regression and lasso to get the best of both worlds. alpha
# is the mixing parameter between ridge (alpha=0) and lasso (alpha=1). That is,
# for Elastic Net there are two parameters to tune: lambda and alpha.

# lets try 25 possible alpha values:
tg <- expand.grid(
  alpha = seq(0, 1, length.out = 15), # [0, 1]
  lambda = 2 ^ seq(10,-2, length = 50) # [0, Inf] SAME lambdas
) 
tg # 15 alphas * 50 lambda = 750 models (each one with 10-fold CV!)

# Train the model:
elastic_fit <- train(
  x = rec,
  data = Hitters.train,
  method = "glmnet",
  tuneGrid = tg,
  trControl =  tc
)
elastic_fit
# yes... that's quite long...


# the best tuning parameter:
elastic_fit$bestTune
elastic_fit$results[726,]



## Exercise--------------------------------------------------------------

# Use the "U.S. News and World Report’s College Data" dataset ('College' in
# ISLR). this dataset contains 777 observations of US colleges with the
# following variables:

data("College", package = "ISLR")
head(College)

?College
# Lets predict Grad.Rate (Graduation rate) from these 17 variables.


# 1) Split to train and test. use 0.7 for the train data

# 2) Then, use each of the learned methods to answer this task. That is:
#   i.   Best Subset Selection 
#   ii.  Ridge regression
#   iii. Lasso
#   iv.  Elastic net (use the alpha = seq(0, 1, length.out=25))

# Notes for the last 3 methods:
# * choose the same lambda values. see that they are broad enough.
# * How? plot change in RMSE and examine yourself. adjust the values if needed.
# * use 5-folds CV.


# Did the method diverged from each other in their performance on test data
# (look at R2)? Which one preformed best?


