
library(tidymodels)
# library(kknn)
# library(randomForest)

library(DALEX)
library(DALEXtra)



# Regression ------------------------------------------------------------------

## Fit a model ---------------------------------------------
# Let's fit a regression model!

data("Hitters", package = "ISLR")
?ISLR::Hitters

# split the data
set.seed(111)
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
Hitters.test <- testing(splits)

# fit a KNN with K=5
rec <- recipe(Salary ~ ., 
              data = Hitters.train) |> 
  step_naomit(Salary) |> 
  step_dummy(all_factor_predictors()) |> 
  step_normalize(all_numeric_predictors())


knn_spec <- nearest_neighbor(
  mode = "regression", engine = "kknn", 
  neighbors = 5
)

knn_fit <- workflow(preprocessor = rec, spec = knn_spec) |> 
  fit(data = Hitters.train)


## Explain the model -----------------------------------------------
# All the methods we'll be using are model agnostic - that means that they work
# for ANY model that can take ANY data and generate a prediction.
# See:
# https://www.tmwr.org/explain
# https://ema.drwhy.ai/



# We first need to setup an explainer:
knn_xplnr <- explain_tidymodels(knn_fit, label = "KNN (K=5)",
                                data = select(Hitters.train, -Salary),
                                y = Hitters.train$Salary)







### Explain a single prediction ------------------------------
# We will be using SHAP to explain the predictions of the model. SHAP is a
# method that tries to estimate what each predictor contributes to each
# prediction - accounting for whatever interactions or conditional effects it
# might have. The ideas are based on game theory - really interesting stuff.
# Lots of math.
#
# SHAP thus tries to answer the Q: 
#   "if X wasn't in the model at all, how would the predicted Y be affected?"
#
# Limitations:
# https://proceedings.mlr.press/v119/kumar20e.html
#
# Alternatives:
# - Local Interpretable Model-agnostic Explanations (LIME): 
#   https://lime.data-imaginist.com


# For example, if we want to see why Bob Horner get a prediction of 
# 1117 (*1000 = 1,117,000$)
predict(knn_fit, new_data = Hitters.test["-Bob Horner",])


# We can look at his SHAP values:
shap_bob <- predict_parts(knn_xplnr,
                          new_observation = Hitters.test["-Bob Horner",],
                          type = "shap")
plot(shap_bob)
# - The sum of all contributions is the values predicted over the baseline - the
#   null average prediction.
# - The box plots represent the distributions of contributions.
#   See: https://ema.drwhy.ai/shapley.html

plot(shap_bob, max_features = Inf) # get more features
plot(shap_bob, show_boxplots = FALSE) # show only mean SHAP values


# Note that SHAP analysis DOES NOT give us any counterfactual information - we
# don't know that if Bob was in a different Division his salary would have been
# lower - just that the model chose to give him a higher prediction because he's
# in division W.
# In other words - we are explaining the model's predictions, but we are not
# explaining the salary! There is not causal information here, we are NOT in
# "explaining mode", we are still in "prediction mode"!






### Variable importance ----------------------
# We've already seen model-based variable importance methods, available through
# the {vip} package. But there are also model-agnostic methods, such as
# Permutation-based variable importance. See:
# https://ema.drwhy.ai/featureImportance.html

# This method assesses a feature's contribution to a model's predictive accuracy
# by randomly shuffling a predictor's values - breaking its relationship with
# the outcome. The model's performance is then evaluated on this new permuted
# data.
# If the variable has no contribution, permuting it will have litle to no effect
# on the model's performance, while variables with larger contributions will
# lead to larger and larger decrease in performance (e.g., larger RMSEs for
# regression).
# This process is repeated multiple times, and the average performance drop is
# used as the importance score, providing a robust measure of each variable's
# contribution.


# permutation?
vi_perm <- model_parts(knn_xplnr, B = 10, 
                       variables = NULL) # specify to only compute for some
plot(vi_perm, bar_width = 5)
# - The vertical line is the baseline RMSE
# - The horizontal bars are the increase in RMSE
# - Box plots are distribution of loss across permutations

plot(vi_perm, bar_width = 5,
     max_vars = Inf, show_boxplots = FALSE)






### Understand a variables contribution ---------

# Partial dependence plots (PDP) visualize the *marginal* effect of one or more
# features on the predicted outcome. That is, how the prediction is affected by
# changing the value of variable X while all other *are held constant*
# (_ceteris-paribus_). See:
#
# https://ema.drwhy.ai/partialDependenceProfiles.html


pdp_hits <- model_profile(knn_xplnr, variables = "Hits", 
                          # default is to plot results of 100 randomly sampled
                          # observations.
                          N = NULL)
plot(pdp_hits) # average
plot(pdp_hits, geom = "profiles") # each line is a single outcome
plot(pdp_hits, geom = "points", variables = "Hits")
# Note that this is a KNN model - it has no structure, and yet, this plot is
# fairly easy to understand!


pdp_league <- model_profile(knn_xplnr, variables = "League")
plot(pdp_league) # average
plot(pdp_league, geom = "points", variables = "League")


# These plots can also show interactions:
pdp_walks.division <- model_profile(knn_xplnr, variables = "Walks",
                                    groups = "Hits")
plot(pdp_walks.division)


# For continuous moderators in a PDP we need the {marginaleffects} package:
marginaleffects::plot_predictions(
  knn_fit, 
  by = c("Walks", "Years"),
  newdata = marginaleffects::datagrid(
    Walks = unique,
    Years = \(v) as.integer(mean(v) + c(-1, 0, 1) * sd(v)),
    
    grid_type = "counterfactual",
    newdata = Hitters.train
  )
)





# Classification ------------------------------------------------------------

## Fit a model -------------------------------------
# Let's fit a classification model!


data("penguins", package = "palmerpenguins")
?palmerpenguins::penguins

# split the data
set.seed(111)
splits <- initial_split(penguins, prop = 0.7)
penguins.train <- training(splits)
penguins.test <- testing(splits)

# We'll fit a random forest model:
rec <- recipe(species ~ .,
              data = penguins.train) |> 
  step_rm(year, island) |> 
  step_impute_mean(all_numeric_predictors()) |> 
  step_impute_mode(all_factor_predictors())

rf_spec <- rand_forest(
  mode = "classification", engine = "randomForest", 
  mtry = sqrt(.cols() - 1),
  trees = 500,
  min_n = 5
)

rf_fit <- workflow(preprocessor = rec, spec = rf_spec) |> 
  fit(data = penguins.train)



## Explain the model -----------------------------------------------


rf_xplnr <- explain_tidymodels(rf_fit, label = "Random Forest",
                               data = select(penguins.train, -species),
                               y = penguins.train$species)


### Explain a single prediction ------------------------------
# Why does the model think that obs 61 has a high chance of being a Gentoo?
predict(rf_fit, new_data = penguins.test[61,], type = "prob")


# We can look at his SHAP values:
shap_61 <- predict_parts(rf_xplnr,
                         new_observation = penguins.test[61,],
                         type = "shap")
plot(shap_61)
# We get the SHAP values for each class!




### Variable importance ----------------------

# Since we're using a random forest model, we can get model-based variable
# importance:
extract_fit_engine(rf_fit) |> 
  vip::vi_model() |> 
  vip::vip(num_features = 100)


# But we can still use the permutation method (or a multiclass model, the loss
# function is a measure of entropy):
vi_perm <- model_parts(rf_xplnr)
plot(vi_perm, max_vars = Inf)
# This matches the plot above very well!
# What's going on with year / island?





### Understand a variables contribution ---------

model_profile(rf_xplnr, variables = c("bill_length_mm", "body_mass_g")) |> 
  plot() + 
  coord_cartesian(ylim = c(0, 1))
# As far as bill length goes, it seems like smaller bills predict Adelie, larger
# for Chinstrap, with Gentoo somewhere in the middle!


# Let's see if this makes sense...
ggplot(penguins.train, aes(bill_length_mm, body_mass_g, color = species)) + 
  geom_point()




marginaleffects::plot_predictions(
  rf_fit, type = "prob",
  by = c("body_mass_g", "group", "bill_length_mm"),
  newdata = marginaleffects::datagrid(
    body_mass_g = unique,
    bill_length_mm = \(v) range(v, na.rm = TRUE),
    
    grid_type = "counterfactual",
    newdata = penguins.train
  )
) + 
  coord_cartesian(ylim = c(0, 1))


