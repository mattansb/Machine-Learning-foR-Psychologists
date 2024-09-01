
library(tidymodels) 

# https://rsample.tidymodels.org/articles/Common_Patterns.html

# We will use the hotel_rates dataset from the modeldata package
data(hotel_rates, package = "modeldata")
?modeldata::hotel_rates
glimpse(hotel_rates)

# This data has *REPEATED MEASURES* from 118 countries
nlevels(hotel_rates$country)

# This means we need to take care when splitting the data - both for the initial
# split and any CV/bootstrap.



# For this demo, we'll use data from the first 15
hotel_rates <- hotel_rates |> 
  mutate(country = droplevels(country)) |> 
  filter(country %in% levels(country)[1:15])





# Initial split --------------------------------------------------------------

# Typically, we would use 
set.seed(111)
splits <- initial_split(hotel_rates, prop = 2/3)
hr.trainX <- training(splits)
hr.testX <- testing(splits)

# When this would mean we have some information from the test countries in our
# training set:
any(hr.testX$country %in% hr.trainX$country)





# Instead, we will use a grouped split:
set.seed(111)
splits <- group_initial_split(hotel_rates, group = country, prop = 2/3)
hr.train <- training(splits)
hr.test <- testing(splits)


# Non of the countries in the test set appear in the training set!
any(hr.test$country %in% hr.train$country)




# Assessing OOS performance using CV -----------------------------------------
# One way to assess the out-of-sample (oos) performance of a model is to use
# cross-validation. This is done by splitting the data into k-folds, training
# the model on k-1 of the folds and validating on the remaining fold. By doing
# this we can get the mean oos performance, but also the between fold
# variability (as realized in the standard error).


# We will be using a linear regression here, looking at R2 and RMSE, but the
# same principles apply to other models and metrics.
linreg_spec <- linear_reg("regression", engine = "lm")

mset_reg <- metric_set(rsq_trad, rmse)
# We will be using rsq_trad() instead of rsq() because the former is more
# sensitive to the number of predictors in the model. This is because rsq()sues caused by grouped data (that are often masked
# by rsq()).

rec <- recipe(avg_price_per_room ~ ., 
              data = hr.train) |> 
  step_dummy(all_factor_predictors())

linreg_wf <- workflow(preprocessor = rec, spec = linreg_spec)


## Standard CV -----------------------------

set.seed(111)
folds5 <- vfold_cv(hr.train, v = 5)
# Data from each country are now scattered across folds.

linreg_rset <- fit_resamples(linreg_wf, 
                             resamples = folds5, 
                             metrics = mset_reg)


## Grouped CV -----------------------------

set.seed(111)
folds5.g <- group_vfold_cv(hr.train, group = country, v = 5)
# Each fold and has a different set of countries

linreg_rset.g <- fit_resamples(linreg_wf, 
                               resamples = folds5.g, 
                               metrics = mset_reg)



## Compare -------------------------------


bind_rows(
  "Ignored" = collect_metrics(linreg_rset), 
  "Accounted for" = collect_metrics(linreg_rset.g),
  
  .id = "Groups"
)
# When ignoring the grouping the data we both over estimate the models
# performance with CV, but we're also over confident in the out-of-sample
# performance by ignoring the dependency (looks at the difference in the std_err
# of our metrics)!




# True OOS Performance --------------------------------------------------

linreg_fit <- fit(linreg_wf, data = hr.trainX)
linreg_fit.g <- fit(linreg_wf, data = hr.train)



hr.test_predictions <- augment(linreg_fit, hr.test)
hr.test_predictions.g <- augment(linreg_fit.g, hr.test)


hr.test_predictions |> mset_reg(avg_price_per_room, .pred)
hr.test_predictions.g |> mset_reg(avg_price_per_room, .pred) # The TRUE OOS!
# Again - we are over estimating our out of sample performance :(


ggplot(mapping = aes(.pred, avg_price_per_room)) + 
  geom_abline() + 
  geom_point(aes(color = "Ignored"), data = hr.test_predictions) + 
  geom_point(aes(color = "Accounted for"), data = hr.test_predictions.g) + 
  coord_obs_pred() + 
  labs(color = "Groups")
  
