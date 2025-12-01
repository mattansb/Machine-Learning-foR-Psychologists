data("Auto", package = "ISLR")
str(Auto)
# The Auto Dataset contains information about cars.
?ISLR::Auto
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

# We're interested in predicting gas consumption: MPG (miles per gallon)

# (Linear) Regression with Base R --------------------------------------------------

# We'll start with an example of fitting and evaluating a regression model using
# base R. Specifically, we'll fit a *linear regression* model (remember that in
# ML-speak, "regression" is any prediction model with a quantitative outcome.)

## 1) Split the data ----------------------------------------------

# We will TRAIN the model (i.e. fit) on the 70% of the observations randomly
# assigned and TEST the model (i.e. predict and assess performance) on the 30%
# that were left.

# because we will use random sampling we need to set a random seed in order to
# replicate the results
set.seed(20251201)

i <- sample.int(nrow(Auto), size = 0.7 * nrow(Auto))
Auto.train <- Auto[i, ]
Auto.test <- Auto[-i, ]


## 2) Specify the model -------------------------------------------
# i. What is the outcome? What are the predictors?
# ii. Does anything need to be transformed or preprocessed somehow?
# iii. How will the predictors be used to predict the outcome?

# In base R, steps i+ii are typically done with a formula:
mpg ~ origin + scale(weight) * horsepower

# Outcome: mpg
# Predictors: origin, weight, horsepower
# Preprocessing:
# - origin is a factor, so will produce dummy coding
# - weight is standardized
# - adding an interaction between (standardized) weight and horsepower

# We can see that all this happens by using the model.matrix() function:
model.matrix(mpg ~ origin + scale(weight) * horsepower, data = Auto.train) |>
  head(n = 10)


# The manner the predictors will be used to predict the outcome is determined by
# the fitting function used. Here, we want a linear regression, so we will use
# the lm() function:
?lm


## 3) Fitting the model -------------------------------------------
# Fitting, or statistical learning, or training is the process of finding the
# best-fitting model for the data. In the case of linear regression, this means
# finding the coefficients that minimize the sum of squared errors.

# We combine the formula, and data and the fitting function as defined above:
fit <- lm(mpg ~ origin + scale(weight) * horsepower, data = Auto.train)


## 4) Evaluate the model ------------------------------------------
# After fitting the model to the training data set, we can see how well it
# performs on the test set.

# Generate predictions:
Auto.test$mpg_pred <- predict(fit, newdata = Auto.test)


# Plot estimated values vs truth
plot(
  Auto.test$mpg_pred,
  Auto.test$mpg,
  xlab = expression("Estimated:" ~ hat(mpg)),
  ylab = "Truth: mpg"
)
abline(a = 0, b = 1)


# How we assess model performance?
# For regression problems- R-squared, MSE, RMSE, MAE...
c(
  rsq = cor(Auto.test$mpg_pred, Auto.test$mpg)^2,
  rmse = sqrt(mean((Auto.test$mpg - Auto.test$mpg_pred)^2))
)


# Let's do this again, with {tidymodels}.

# (Linear) Regression with {tidymodels}  ----------------------------------------------

# We will use the tidymodels ecosystem:
library(tidymodels)
# The tidymodels ecosystem is a collection of packages for modeling and machine
# learning using tidyverse principles. It includes tools for:
#
# - data splitting
# - data preprocessing and feature engineering
# - model specification and fitting
# - model tuning and evaluation (we will get to this in a few weeks)
#
# (see: https://www.tidymodels.org/)

## 1) Split the data ----------------------------------------------

splits <- initial_split(Auto, prop = 0.7) # create a splits object
splits # see the sizes of the sets
Auto.train <- training(splits) # Extract the training set
Auto.test <- testing(splits) # Extract the test set


## 2) Specify the model -------------------------------------------

### i. Define outcome + predictors --------------------

# We again use a formula, but inside the recipe()
rec <- recipe(mpg ~ origin + weight + horsepower, data = Auto.train)
rec


### ii. Preprocessing ---------------------------------

# This is done by adding "steps" to the recipe - how should variables be
# processed *prior* to the model fitting.

# There are many prepossessing "steps" we can take:
# https://recipes.tidymodels.org/reference/index.html
# For example, here we want:
rec <- rec |>
  # - origin is a factor, so will produce dummy coding
  step_dummy(origin) |>
  # - weight is standardized
  step_normalize(weight) |>
  # - adding an interaction between (standardized) weight and horsepower
  step_interact(~ weight:horsepower)

# This is quite verbose compared to the formula, but will come in handy with
# more complicated steps and models.

rec

# Right now, the recipe is just a list of general instructions. To get a recipe
# with specific instruction steps, we need to train the recipe. In the
# tidymodels terminology, the process of training a recipe is called
# "preparing":
prep(rec)

# We can then use the "prepared" recipe to "bake" some data into the shape we
# want:
prep(rec) |> bake(new_data = Auto.train)
# Compare this to the model matrix above

### iii. Define the type of model ---------------------

# In {tidymodels} this stage is separate from the model fitting. We define a
# model specification (or model "spec") using one of the functions from the
# {parsnip} package (or one if its extensions).
linreg_spec <- linear_reg(mode = "regression", engine = "lm")

# This spec will then we "translated" to use the correct underlying fitting
# function:
translate(linreg_spec)


# There are many different ways to fit a linear regression - we will learn some
# of them over the coming weeks. These different ways are controlled by
# different "engines" that can be used to fit the same model spec.
show_engines("linear_reg")


# Finally, we can combine the recipe and the model spec to a workflow - together
# they tell us how data *should* be used to fit a model.
linreg_wf <- workflow(preprocessor = rec, spec = linreg_spec)
linreg_wf


## 3) Fitting the model -------------------------------------------

# Fitting the model is as easy as passing the workflow and some training data
# to the fit() function:
linreg_fit <- fit(linreg_wf, data = Auto.train)

# We can extract the underlying model object:
linreg_engine <- extract_fit_engine(linreg_fit)
class(linreg_engine)
# compare to:
cbind(
  "{tidymodels}" = coef(linreg_engine),
  "stats::lm" = coef(fit)[c(1, 4, 5, 2, 3, 6)]
)
# (Why aren't these exactly the same? How is this related to bias or variance?)

## 4) Evaluate the model ------------------------------------------

# Generate predictions:
predict(linreg_fit, new_data = Auto.test) # generates a data frame
predict(linreg_fit, new_data = Auto.test, type = "raw") # generates a vector

# Or we can "augment" a dataset - add predictions to it
Auto.test_predictions <- augment(linreg_fit, new_data = Auto.test)
head(Auto.test_predictions)

# In either case, the test set is preprocessed according to the recipe, and
# predictions are then made.

ggplot(Auto.test_predictions, aes(.pred, mpg)) +
  geom_abline() +
  geom_point() +
  coord_obs_pred() +
  labs(x = expression("Estimated:" ~ hat(mpg)), y = "Truth: mpg")

# Performance metrics
Auto.test_predictions |> rsq(mpg, .pred)
Auto.test_predictions |> rmse(mpg, .pred)

# or define a metric est:
mset_reg <- metric_set(rsq, rmse, mae)
Auto.test_predictions |> mset_reg(mpg, .pred)
