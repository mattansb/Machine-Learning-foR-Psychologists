

library(tidymodels)


# The data and problem ----------------------------------------------------

# Previously, we've used {tidymodels} for a regression problem. Today we are looking
# at classification.

# Smarket dataset contains daily percentage returns for the S&P 500 stock index
# between 2001 and 2005 (1,250 days).
data(Smarket, package = "ISLR")
?ISLR::Smarket
# For each date, the following vars were recorded:
# - Lag1--Lag5 - percentage returns for each of the five previous trading days.
# - Volume - the number of shares traded on the previous day(in billions).
# - Today - the percentage return on the date in question.
# - Direction - whether the market was Up or Down on this date.

# Assume the following classification task on the Smarket data:
# predict Direction (Up/Down) using the features Lag1 and Lag2.
# If we are not sure how Direction is coded we can use levels():
levels(Smarket$Direction)

table(Smarket$Direction)
# The base rate probability:
table(Smarket$Direction) |> proportions()


# Data Splitting (70%):
set.seed(1234)
splits <- initial_split(Smarket, prop = 0.7)
Smarket.train <- training(splits)
Smarket.test <- testing(splits)


# We'll start by using a parametric method - logistic regression.


# A logistic regression (with tidymodels) ---------------------------------

## 1) Specify the model -------------------------------------------

# There are several engines for a logistic regression:
show_engines("logistic_reg")

# We'll use the standard stats::glm()
logit_spec <- logistic_reg(mode = "classification", engine = "glm")

# We can see the under the hood, stats::glm() is used:
translate(logit_spec)




## 2) Feature Preprocessing -------------------------------------------

rec <- recipe(Direction ~ Lag1 + Lag2 + Lag3, 
              data = Smarket.train)
# Logistic regression does not require any preprocessing.


# Combine spec and recipe to a workflow:
logit_wf <- workflow(preprocessor = rec,
                     spec = logit_spec)
logit_wf




## 3) Fit the model ---------------------------------------------------

logit_fit <- fit(logit_wf, data = Smarket.train)

extract_fit_engine(logit_fit) |> 
  parameters::model_parameters(exponentiate = TRUE)
# Or...





## 4) Predict and evaluate -------------------------------------------------

predict(logit_fit, new_data = Smarket.test, type = "class") # default
predict(logit_fit, new_data = Smarket.test, type = "prob")

Smarket.test_predictions <- augment(logit_fit, new_data = Smarket.test)
head(Smarket.test_predictions)
# The .pred_Down and .pred_Up give the probabilistic predictions for each class.
# They add up to 1 for each row.



Smarket.test_predictions |> 
  conf_mat(truth = Direction, estimate = .pred_class)


mset_classifier <- metric_set(accuracy, sensitivity, specificity, f_meas)

Smarket.test_predictions |> 
  mset_classifier(truth = Direction, estimate = .pred_class, 
                  # Tell the function that the positive class is "Up"
                  event_level = "second")
# Overall, not amazing...


# Since this is a probabilistic model, we can also look at the ROC curve and AUC:
Smarket.test_predictions |> 
  roc_curve(truth = Direction, .pred_Up, event_level = "second") |> 
  autoplot()

# And indeed...
Smarket.test_predictions |> 
  roc_auc(truth = Direction, .pred_Up, event_level = "second")


# Can we use KNN to get better predictions?

