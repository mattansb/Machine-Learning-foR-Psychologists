                  ### Forward/Backward/Stepwise selection ###

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


rec <- recipe(Salary ~ ., data = Hitters_train) |> 
  # Standardize numeric variables:
  step_center(all_numeric_predictors()) |> 
  step_scale(all_numeric_predictors()) |> 
  # Make one hot encoding dummy variables
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


# Forward Selection --------------------------------------

tg <- expand.grid(nvmax = 1:19)

tc <- trainControl(method = "cv", number = 5)

fit.fwd <- train(
  x = rec,
  data = Hitters,
  method = "leapForward",
  tuneGrid = tg,
  trControl = tc
)

fit.fwd
plot(fit.fwd)
fit.fwd$bestTune # gives the row number
coef(fit.fwd$finalModel, id = 14)

# Backward Selection --------------------------------------

fit.bwd <- train(
  x = rec,
  data = Hitters,
  method = "leapBackward",
  tuneGrid = tg,
  trControl = tc
)

plot(fit.bwd)

# Step wise ----------------------------------------------

fit.sw <- train(
  x = rec,
  data = Hitters,
  method = "leapSeq",
  tuneGrid = tg,
  trControl = tc
)

plot(fit.sw)


# Compare ---------------------------------------------------

# using best-subset\ forward\ backward selection, can lead to different results.

fit.bwd$bestTune$nvmax # And 11 when using Forward
fit.fwd$bestTune$nvmax # Best is 14 when using Backward
fit.sw$bestTune$nvmax # And 11 for stepwise...


# Predict using test data.
Hitters_test$pred.bwd <- predict(fit.bwd, newdata = Hitters_test)
Hitters_test$pred.fwd <- predict(fit.fwd, newdata = Hitters_test)
Hitters_test$pred.sw <- predict(fit.sw, newdata = Hitters_test)  

# Assess the test error for each type of models:
rsq(Hitters_test, truth = Salary, estimate = pred.bwd)
rsq(Hitters_test, truth = Salary, estimate = pred.fwd)
rsq(Hitters_test, truth = Salary, estimate = pred.sw)

# Other.... ----------------------------

# See also
?MASS::stepAIC

