### Model Selection ###

library(tidymodels)

library(leaps)

# Recommendation: read "Feature Selection Overview" on caret website:
# https://topepo.github.io/caret/feature-selection-overview.html


# Hitters DATA and the PROBLEM ---------------------------------
# (a REGRESSION problem)

# Hitters Dataset: Baseball Data from the 1986 and 1987 seasons
data("Hitters", package = "ISLR")
ISLR::Hitters
Hitters <- tidyr::drop_na(Hitters, Salary)
# A data frame with 263 observations of major league players on the following 20
# variables.

dim(Hitters)
names(Hitters)


# We wish to predict a baseball player's *Salary* on the basis of preformence
# variables in the previous year.
# Which 19 predictors will be best for predicting Salary?


# Split:
set.seed(123442) 
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
Hitters.test <- testing(splits)
# Our data is REALLY SMALL such that splitting the data to train and test might
# leave us with very small datasets.



# Best Subset Selection Method --------------------------------------

# We will see how to fit Best Subset Selection. Best Subset Selection is
# generally better than the stepwise/forward/backward methods since it is less
# biased.

# We can use the {leaps} package to preform Best Subset Selection based on Cp,
# adj-R2, or BIC.

regfit.full <- regsubsets(Salary ~ ., data = Hitters.train)
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
  pivot_longer(cols = -nv, 
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


## Assess performance on test set ---------------------------------

# the leaps package doesn't provide a predict method for `regsubsets` objects.
# Se we will define one ourselves
predict.regsubsets <- function (object, newdata, id = NULL, 
                                select = c("adjr2", "cp", "bic")) {
  cl <- object$call
  cl[[1]] <- quote(stats::lm)
  cl[!names(cl) %in% c(formalArgs(stats::lm), "")] <- NULL
  lm_object <- eval.parent(cl)
  
  X_newdata <- model.matrix(terms(lm_object), newdata, 
                            contrasts.arg = object$contrasts)
  
  if (is.null(id)) {
    select <- match.arg(select)
    v <- summary(object)[[select]]
    id <- switch (select,
                  adjr2 = which.max(v),
                  cp = ,
                  bic = which.min(v)
    )
  }
  
  b <- coef(object, id = id)
  
  as.vector(X_newdata[, names(b), drop = FALSE] %*% b)
}


Hitters.test$pred_bss <- predict(regfit.full, newdata = Hitters.test, id = 9) 

Hitters.test |> rsq(truth = Salary, estimate = pred_bss)
# Not bad!


# Stepwise -----------------------------------------------------------------

# First, fit the full model
regfit.all <- lm(Salary ~ ., data = Hitters.train)

?MASS::stepAIC
# Selection is based on AIC (similar to Cp or BIC), and supports a wide range
# of model types.

## Forward -------------------------------

regfit.fwd <- MASS::stepAIC(regfit.all, direction = "forward")

## Backward -------------------------------

regfit.bwd <- MASS::stepAIC(regfit.all, direction = "backward")

## Both -------------------------------

regfit.both <- MASS::stepAIC(regfit.all, direction = "both")


# Compare ---------------------------------------------------

# using best-subset\ forward\ backward selection, can lead to different results.

length(insight::find_terms(regfit.fwd)[["conditional"]]) # 19 (forward from max...)
length(insight::find_terms(regfit.bwd)[["conditional"]]) # 9
length(insight::find_terms(regfit.both)[["conditional"]]) # 9


# Predict using test data.
Hitters.test$pred_bwd <- predict(regfit.fwd, newdata = Hitters.test)
Hitters.test$pred_fwd <- predict(regfit.bwd, newdata = Hitters.test)
Hitters.test$pred_both <- predict(regfit.both, newdata = Hitters.test)  

# Assess the test fit for each type of models:
Hitters.test |> rsq(truth = Salary, estimate = pred_bss)
Hitters.test |> rsq(truth = Salary, estimate = pred_bwd)
Hitters.test |> rsq(truth = Salary, estimate = pred_fwd)
Hitters.test |> rsq(truth = Salary, estimate = pred_both)



