                  ### SVM ###

library(ggplot2)

library(rsample)
library(recipes)
library(caret)
library(yardstick)




# The OJ Data -----------------------------------------------------------------

data("OJ", package = "ISLR")
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
  step_range(all_numeric_predictors())
# The decision boundary maximizes the distance to the nearest data points from
# different classes. Hence, the distance between data points affects the
# decision boundary SVM chooses. In other words, training an SVM over the scaled
# and non-scaled data leads to the generation of different models.


tc <- trainControl(method = "cv", number = 5)

# Support Vector Classifier --------------------------------------------------
# FITTING the support vector classifier (method = "svmLinear")


## Train ------------------------

# We use the train() function to fit the support vector classifier for a given
# value of the cost parameter. Here, we directly try more then one value and
# find the best fit with CV!

tg <- expand.grid(
  C = 10 ^ seq(-3, 0, length = 20) # [0, 1]
)
tg
# A cost argument allows us to specify the cost of a violation to the margin:
# small cost -> wide margins and many support vectors violate the margin.
# large cost -> narrow margins and few support vectors violate the margin.

set.seed(1) # for CV
fit.lin <- train(
  x = rec,
  data = OJ.train,
  method = "svmLinear",
  tuneGrid = tg,
  trControl  = tc
)


fit.lin # Best fit with cost = 0.335

# Plot CP by accuracy:
plot(fit.lin, xTrans = log)


## Explore the final model ----------------------

fit.lin$finalModel
# best training error is 0.169393 (for the cost 0.335)
# Number of Support Vectors : 365! 


# Here we demonstrated the use of this function on a two-dimensional example so
# that we can plot the resulting decision boundary.
X_train <- bake(prep(rec), new_data = NULL, all_predictors(), composition = "matrix")
plot(fit.lin$finalModel, data = X_train)


# Full circles\triangles = support vectors (365 obs.)
# Hollow circles\triangles = the remaining observations 



## Evaluating the support vector classifier performance ---------------------
# (on a new TEST DATA)

## Predict the class labels of these test observations. 
# We use the best model obtained through CV:

OJ.test$pred_lin <- predict(fit.lin, newdata = OJ.test)

confusionMatrix(OJ.test$pred_lin, OJ.test$Purchase) # confusion matrix
#>                Accuracy : 0.8364
#>             Sensitivity : 0.9281
#>             Specificity : 0.6667







# Support Vector Machine ----------------------------------------
# (non-linear kernel)


## Polynomial ------------------------------------

tg <- expand.grid(
  C = 10 ^ seq(-3, 0, length = 20), # [0, 1]
  degree = 2, # [1, Inf]
  scale = 1 # [0, Inf]
)
# Here we also add the degree argument to specify a degree for the polynomial
# kernel (for quadratic: degree = 2).

# We will also add a parameter called "scale" which is related to gamma\sigma
# which will become relevant for radial kernel, but means nothing for poly
# kernel. For some reason caret insist we will set some value (non zero) in
# scale. Without it train() will not run. We will put scale = 1, but it means
# nothing!


set.seed(1) 
fit.poly2 <- train(
  x = rec,
  data = OJ.train,
  method = "svmPoly",
  tuneGrid = tg,
  trControl  = tc
)


fit.poly2$bestTune
plot(fit.poly2, xTran = log)


plot(fit.poly2$finalModel, data = X_train)


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


tg <- expand.grid(
  C = 10 ^ seq(-3, 0, length = 20), # [0, 1]
  sigma = 2 ^ seq(-2, 2, length = 5) # [0, Inf]
)

set.seed(1)
fit.rad <- train(
  x = rec,
  data = OJ.train,
  method = "svmRadial",
  tuneGrid = tg,
  trControl  = tc
)


fit.rad$bestTune
plot(fit.rad, xTran = log)


plot(fit.rad$finalModel, data = X_train)


## Compare models ----------------------------------------------------------


OJ.test$pred_poly2 <- predict(fit.poly2, newdata = OJ.test)
OJ.test$pred_rad <- predict(fit.rad, newdata = OJ.test)


accuracy(OJ.test, truth = Purchase, estimate = pred_lin)
accuracy(OJ.test, truth = Purchase, estimate = pred_poly2)
accuracy(OJ.test, truth = Purchase, estimate = pred_rad)







# SVM with Multiple Classes -------------------------------------------------

data("Wage", package = "ISLR")

Wage$maritl3 <- forcats::fct_collapse(
  Wage$maritl, 
  "3. no longer married" = c("3. Widowed", "4. Divorced", "5. Separated")
)

set.seed(1)
splits <- initial_split(Wage, prop = 0.7)
Wage.train <- training(splits)
Wage.test <- testing(splits)


# We will try to predict marital status from age and wage.

table(Wage.train$maritl3)
table(Wage.train$maritl3) |> proportions()
# If the response is a factor containing more than two levels, then train()
# will perform multi-class classification using the one-versus-one approach.


rec <- recipe(maritl3 ~ age + wage,
              data = Wage.train) |> 
  step_range(all_numeric_predictors())


tg <- expand.grid(
  C = 10 ^ seq(-3, 0, length = 10) # [0, 1]
)

set.seed(1)
fit.lin3class <- train(
  x = rec,
  data = Wage.train,
  method = "svmLinear",
  tuneGrid = tg,
  trControl  = tc
)


fit.lin3class$bestTune


## Predict and evaluate

Wage.test$pred_lin <- predict(fit.lin3class, newdata = Wage.test)

# 3 classes-confusion matrix:
confusionMatrix(Wage.test$pred_lin,Wage.test$maritl3) 

# Is this surprising? Not really, considering our predictors..
ggplot(Wage.train, aes(age, wage)) +
  geom_point(aes(color = maritl3))


# SVR --------------------------------------------------------------------

# both caret and e1071 libraries can also perform support vector regression, if
# the response vector that is numerical rather than a factor.



rec <- recipe(wage ~ age + maritl3, 
              data = Wage.train) |> 
  step_range(all_numeric_predictors()) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# Note that for SVR, the C parameter can be bigger then 1 - it represents the
# total allowed error.

tg <- expand.grid(
  C = 10 ^ seq(-3, 2, length = 20) # [0, Inf]
)
# A cost argument allows us to specify the cost of a violation to the margin:
# small cost -> wide margins and many support vectors violate the margin.
# large cost -> narrow margins and few support vectors violate the margin.

set.seed(1) # for CV
fit.lin2 <- train(
  x = rec,
  data = Wage.train,
  method = "svmLinear",
  tuneGrid = tg,
  trControl  = tc
)

fit.lin2$bestTune


# Exercise ---------------------------------------------------------------

# 1. Take one of the datasets used here (Wage / OJ) and predict the outcome with
#    4 variables of your choice.
# 2. Use two method (linear, poly, radial) and compare their performance.



