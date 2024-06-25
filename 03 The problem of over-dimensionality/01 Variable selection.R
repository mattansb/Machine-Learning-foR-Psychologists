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
# Our data is REALLY SMALL such that splitting the data to train and test might
# leave us with very small datasets.



# PART A: Model Selection - Best Subset Selection Method ----------------------

# We will see how to fit Best Subset Selection. Best Subset Selection is
# generally better than the stepwise/forward/backward methods (script 03) - it
# is less biased.

# We can use the {leaps} package to preform Best Subset Selection based on Cp,
# adj-R2, or BIC.

regfit.full <- leaps::regsubsets(Salary ~ ., data = Hitters.train)
summary(regfit.full)
# Outputs the best set of variables for each model size up to the best
# 8-variable model (8 is the default).
# An asterisk indicates the variable is included in the corresponding model. 


# If we want we can fit in this data up to a 19-variable model (and not 8) using
# the nvmax option.
regfit.full <- regsubsets(Salary ~ ., data = Hitters.train, nvmax = 19)
summary(regfit.full)

# Let's get a closer look on the statistics of this output:
reg.summary <- summary(regfit.full)
names(reg.summary) 
# The summary() function also returns R2 (rsq), RSS, adjusted R2 (adjr2), Cp,
# and BIC. We can examine these statistics to select the best overall model.
reg.summary_df <- data.frame(nv = 1:19, reg.summary[2:6]) |> 
  tidyr::pivot_longer(cols = -nv, 
                      names_to = "Index", 
                      values_to = "value")

ggplot(reg.summary_df, aes(nv, value)) +
  facet_wrap(~Index, scales = "free", ncol = 3) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:19, minor_breaks = NULL)
# For instance, we see that R2 increases from 32% for 1-variable model, to
# almost 55%, for 19-variables model. As expected, the R2 increases
# monotonically as more variables are included - but this is not the case for
# adjusted R2.
# Based on BIC, we would select a 5-var model.
coef(regfit.full, id = 5)
# Based on Adj. R2 and on Cp, we would select a 9-var model.
coef(regfit.full, id = 9)



## Tuning the nvmax parameter using resampling -------------------------------
# Instead of selecting the optimal number of variables using out-of-sample
# performance approximates, we can use resampling methods.

# Unfortunately {caret} doesn't natively support best subset selection...
# We need to help it a little:
source(".caret-leapExhaustive.R")

## (A) Prepossessing
rec <- recipe(Salary ~ ., data = Hitters.train) |> 
  step_dummy(all_factor_predictors())


## (B) Tune:
# One hyperparameter - nvmax: Maximum Number of Predictors to concider.
tg <- expand.grid(nvmax = 1:19)


# Fit ridge regression with 10-folds cv:
tc <- trainControl(method = "cv", number = 10)

set.seed(1)
bestsub_fit <- train(
  x = rec,
  data = Hitters.train,
  method = leapExhaustive, # note we are passing the object, not a name
  tuneGrid = tg,
  trControl =  tc
)

plot(bestsub_fit) # best fit, based on CV RMSE, has 5 predictors.
# In the final model, those 5 predictors are:
coef(bestsub_fit$finalModel, id = bestsub_fit$bestTune$nvmax)



# (C) Predict:
Hitters.test$bss.pred <- predict(bestsub_fit, newdata = Hitters.test)

# EVALUATE the RMSE on the TEST set, associated with this value of lambda:
rsq(Hitters.test, truth = Salary, estimate = bss.pred)
rmse(Hitters.test, truth = Salary, estimate = bss.pred)
hist(Hitters.train$Salary)


# PART B - Regularization: Shrinkage Methods ----------------------

## Ridge Regression -------------------------

# We will perform ridge regression and the lasso in order to predict Salary on
# the Hitters data.

## (A) Prepossessing
rec <- rec |> 
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
plot(rigreg_fit)
# a little hard to see... 
ggplot(rigreg_fit) +
  scale_x_continuous(trans = scales::transform_log(),
                     breaks = scales::breaks_log())
(bestlambda <- rigreg_fit$bestTune$lambda)



# (C) Predict:
Hitters.test$ridge.pred <- predict(rigreg_fit, newdata = Hitters.test)
# Uses the best lambda

# EVALUATE the RMSE on the TEST set, associated with this value of lambda:
rsq(Hitters.test, truth = Salary, estimate = ridge.pred)
rmse(Hitters.test, truth = Salary, estimate = ridge.pred)


# (D) Coefficients:
## We can extract the model's coefficients according to lambda:
# The coef of the chosen model:
coef(rigreg_fit$finalModel, s = bestlambda)   
# or other lambdas...
# E.g. for lambda = 0.0000, (this result should be similar to OLS result)
coef(rigreg_fit$finalModel, s = 0)


# the different models produced by different lambdas! and the parameters gets
# smaller as lambda rises:
plot(coef(rigreg_fit$finalModel, s = 0), ylab = "Coef"); abline(0, 0)
plot(coef(rigreg_fit$finalModel, s = bestlambda), ylab = "Coef"); abline(0, 0)
plot(coef(rigreg_fit$finalModel, s = 10000), ylab = "Coef"); abline(0, 0)

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

tg$alpha <- 1
tg

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
plot(coef(lasso_fit$finalModel, s = 0), ylab = "Coef"); abline(0, 0)
plot(coef(lasso_fit$finalModel, s = 10), ylab = "Coef"); abline(0, 0)
plot(coef(lasso_fit$finalModel, s = 100), ylab = "Coef"); abline(0, 0)

# the best tuning parameter:
ggplot(lasso_fit) +
  scale_x_continuous(trans = scales::transform_log(),
                     breaks = scales::breaks_log())
(bestlambda <- lasso_fit$bestTune$lambda)


# (C) PREDICTING for test data (using best lambda).
Hitters.test$lasso.pred <- predict(lasso_fit, newdata = Hitters.test) 

rmse(Hitters.test, truth = Salary, estimate = lasso.pred)
rsq(Hitters.test, truth = Salary, estimate = lasso.pred)
# for ridge it was slightly smaller, but this one is much more parsimonious:
# 7 predictors instead of 19!
(coef <- coef(lasso_fit$finalModel, s = bestlambda))
sum(coef==0) # some coefs are exactly 0!
plot(coef); abline(0, 0)


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
tg # 15 alphas * 50 lambda = 750 models (each one with 5-fold CV!)

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
rownames(elastic_fit$bestTune)
elastic_fit$results[rownames(elastic_fit$bestTune),]



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

# Notes for the last 3 methods, you should use the same lambda values - make
# sure they are broad enough to capture a desired RMSE minima. You can do this
# by plotting RMSE vs lambda and see if there is a "valley". 
# Use 5-folds CV to ture alpha/almbda.

# 3) Compare:
# - Did the method diverged from each other in their performance on test data
#   (look at R2)? Which one preformed best on the test set?
# - Compare the Best Subset Selection, LASSO and Elastic net - did they all
#   "choose" similar predictors?


