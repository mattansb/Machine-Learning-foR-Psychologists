
library(patchwork)

library(tidymodels)
# library(kknn)


# The data and problem ----------------------------------------------------


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

# But this time we won't be using a linear regression model - we're using KNN!


# 1) Split the data ----------------------------------------------

splits <- initial_split(Auto, prop = 0.7) # create a splits object
Auto.train <- training(splits) # Extract the training set
Auto.test <- testing(splits) # Extract the test set




# 2) Specify the model -------------------------------------------

## Model specification ---------------------


knn_spec <- nearest_neighbor(
  mode = "regression", engine = "kknn", 
  neighbors = 5
)

# KNN only has one engine:
show_engines("nearest_neighbor")

# This spec will then we "translated" to use the correct underlying fitting
# function:
translate(knn_spec)




## Define outcome + predictors + preprocessing --------------------

# Since KNN identifies neighbors of observations according to their
# **distance**, the scale of the variables matters: large scale -> larger
# distance between the observations on that X.
# So we need to re-scale all variables. And we also need to dummy code our
# factor (origin).
# 
# You can see the _required_ preprocessing steps in the model spec details:
?details_nearest_neighbor_kknn
# ?details_{spec}_{engine}


# There are several ways to do this -
# here's one.

rec <- recipe(mpg ~ origin + weight + horsepower,
              data = Auto.train) |> 
  step_dummy(origin) |> 
  # The Yeo–Johnson transformation (a generalization of the Box-Cox
  # transformation) can be used to make highly skewed variables resemble a more
  # normal-like distribution, typically improving the performance of the model.
  # https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation
  # It requires the "training" of a Lambda parameter, which {recipes} finds for
  # us.
  step_YeoJohnson(horsepower) |> 
  step_normalize(all_numeric())
# Note that the ORDER OF STEPS - where we put step_dummy() determines if
# the dummies will be centered and scaled!
rec


(ggplot(Auto.train, aes(horsepower)) + 
    geom_density() + 
    labs(title = "Raw")) + 
  (bake(prep(rec), new_data = NULL) |> 
     ggplot(aes(horsepower)) + 
     geom_density() + 
     labs(title = "Standardized Yeo–Johnson"))


# Finally, we can combine the recipe and the model spec to a workflow - together
# they tell us how data *should* be used to fit a model.
knn_wf <- workflow(preprocessor = rec, spec = knn_spec)
knn_wf




## 3) Fitting the model -------------------------------------------

# Fitting the model is as easy as passing the workflow and some training data
# to the fit() function:
knn_fit <- fit(knn_wf, data = Auto.train)




## 4) Predict and Evaluate the model -------------------------------

# Generate predictions:
Auto.test_predictions <- augment(knn_fit, new_data = Auto.test)
head(Auto.test_predictions)

# In either case, the test set is preprocessed according to the recipe, and
# predictions are then made.

ggplot(Auto.test_predictions, aes(.pred, mpg)) + 
  geom_abline() + 
  geom_point() + 
  coord_obs_pred() + 
  labs(x = expression("Estimated:"~hat(mpg)), 
       y = "Truth: mpg")
# What happened here?? Got back and fix it...

# Performance metrics
mset_reg <- metric_set(rsq, rmse, mae)
Auto.test_predictions |> mset_reg(mpg, .pred)


# Compare this to performance on the training set:
augment(knn_fit, new_data = Auto.train) |> mset_reg(mpg, .pred)




# Exercise ---------------------------------------------------------------

# 1. Define a recipe for a new KNN model
#   Add the required steps for KNN.
rec2 <- recipe(mpg ~ ., # all predictors,
               data = train.data) |>
  # Remove the {name} predictor
  step_rm(name) 


# 2. Fit a KNN model with k=5


# 3. Evaluate the new model on the test set.
#   How does it compare to the k=5 model with 3 predictors?


# 4. Repeat steps 2 and 3 with k=10. What can we expect to happen?


