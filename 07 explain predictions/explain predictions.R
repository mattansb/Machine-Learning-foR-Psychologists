
library(tidymodels)
# library(kknn)
# library(randomForest)

library(kernelshap)
library(shapviz)


# Regression ------------------------------------------------------------------
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


## Build SHAP explainer ------------------------------
# We will be using SHAP to explain the predictions of the model. SHAP are model
# agnostic - this mean that we can use them to explain any model, not just the
# ones that have a closed form solution (like linear regression). SHAP is a
# method that# tries to estimate what each predictor contributes to each
# prediction - accounting for whatever interactions or conditional effects it
# might have. The ideas are based on game theory - really interesting stuff.
# Lots of math.
#
# SHAP thus tries to answer the Q: 
#   "if X wasn't in the model at all, how would Y be affected?"
# 
# https://modeloriented.github.io/kernelshap
# https://modeloriented.github.io/shapviz
#
# Limitations:
# https://proceedings.mlr.press/v119/kumar20e.html
#
# Alternatives:
# https://lime.data-imaginist.com
# https://koalaverse.github.io/vip
# https://modeloriented.github.io/DALEX
# https://modeloriented.github.io/DALEX/reference/model_profile.html


# The steps:
# 1. Train the explainer with the model and the training data (X_bg)
# 2. Use the explainer to explain predictions.
shaps <- kernelshap(knn_fit, 
                    # The test set, whose predictions we're trying to explain.
                    # Only include the predictors!
                    X = Hitters.test |> select(-Salary), 
                    # The training data. Note that if the training set it very
                    # large (N>500) you can use a subset of the data.
                    bg_X = Hitters.train)
# This took about 5 minutes on my computer, but depending on the model and the
# data it can easily take hours.

shaps <- readRDS("knn_shaps.rds")


# 3. Explore the explanations:
sv <- shapviz(shaps)
# We're using {shapvis} to visualize the SHAP results



## Explain a single observation -------------
# We can also look at single predictions. For example, if we want to 
# see why Bob Horner get a prediction of 1117 (*1000 = 1,117,000$)
predict(knn_fit, new_data = Hitters.test["-Bob Horner",])
which(rownames(Hitters.test) == "-Bob Horner")
# We can look at his SHAP values:

sv_waterfall(sv, row_id = 10)
sv_waterfall(sv, row_id = 10, max_display = Inf)


# Or
sv_force(sv, row_id = 10, max_display = 10)
# Note that SHAP analysis DOES NOT give us any counterfactual information - we
# don't know that if he was in a different Division his salary would have been
# lower - just that the model chose to give him a higher prediction because he's
# in division W.
# In other words - we are explaining the model's predictions, but we are not
# explaining the salary! There is not causal information here, we are NOT in
# "explaining mode", we are still in "prediction mode"!



## Variable importance ----------------------

sv_importance(sv, show_numbers = TRUE)
# By averaging the *absolute* SHAP values, we can see which variables contribute 
# more, overall, relative to the others. 
# This gives a global view of the model.



## Explain a variables contribution ---------

sv_dependence(sv, v = "Division", color_var = NULL)
# Each point is an observation in the test set.
# We can see that being in Division W tends to produce predictions of a lower
# salary compared to being in Division E.



sv_dependence(sv, v = "Hits", color_var = NULL)
# Looking at Hits, we can see an almost linear trend (note the KNN is
# none-parametric!)

sv_dependence(sv, v = "Hits", color_var = NULL) + 
  # we can even add a smooth visualize the trend.
  geom_smooth(se = FALSE)


# By default, `sv_dependence` colors the dots according to a variable it has
# automagically deemed to have the strongest interaction with the focal
# predictor. Setting `color_var = NULL` removes this (recommended). But we can
# also set this manually, if we wanted.

 

# We can also look at a bivariate plot:
sv_dependence2D(sv, x = "Hits", y = "HmRun")







# For more advance plots, we can use this handy little function I cooked up:
# This function takes a shapviz object and converts it to a data frame that can
# be manipulated / plotted with ggplot.
extract_agg_shaps <- function(x, variables, ...) {
  UseMethod("extract_agg_shaps")
}

extract_agg_shaps.shapviz <- function(x, variables, ...) {
  out <- x[["X"]][,variables, drop = FALSE]
  out[[".shap"]] <- rowSums(x[["S"]][,variables, drop = FALSE])
  if (!is.null(rownames(out))) {
    out <- tibble::rownames_to_column(out, var = ".row")
  } else {
    out <- tibble::rowid_to_column(out, var = ".row")
  }
  out
}

extract_agg_shaps.mshapviz <- function(x, variables, ...) {
  lapply(x, extract_agg_shaps, variables = variables) |> 
    dplyr::bind_rows(.id = ".class")
}


extract_agg_shaps(sv, variables = c("Hits", "HmRun", "Division")) |> 
  # SHAP values are saved in the ".shap" column. 
  ggplot(aes(Hits, .shap, color = Division)) + 
  facet_grid(cols = vars(HmRun = cut_number(HmRun, 3)), 
             labeller = label_both) + 
  geom_point() + 
  geom_smooth(se = FALSE, method = "gam")











# Classification ------------------------------------------------------------
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




## Build SHAP explainer ------------------------------
# It's basically the same - but we have an explainer for each class!


shaps <- kernelshap(rf_fit, 
                    X = penguins.test, 
                    
                    # Only include the predictors! Here is an alternative way to
                    # do this:
                    bg_X = penguins.train,
                    feature_names = c("bill_length_mm", "bill_depth_mm",
                                      "flipper_length_mm", "body_mass_g", "sex"),
                    
                    # We want to explain the predicted probabilities, se we need
                    # to set `type = "prob"`. (It the model was non
                    # probabilistic we would omit this argument.)
                    type = "prob")
# We need to tell the model to predict probabilities


# 3. Explore the explanations:
sv <- shapviz(shaps)




## Explain a single observation -------------
# Why does the model think that obs 61 has a high chance of being a Gentoo?
predict(rf_fit, new_data = penguins.test[61,], type = "prob")
# We can look at his SHAP values:

sv_waterfall(sv, row_id = 61)
# We can also ask for a single class but subsetting the shapviz:
# (This is true for all the following functions as well)
sv_waterfall(sv$.pred_Gentoo, row_id = 61) 


# Or
sv_force(sv, row_id = 61)


## Variable importance ----------------------

sv_importance(sv, show_numbers = TRUE)
# We can see that for different classes, different features are important.



## Explain a variables contribution ---------


sv_dependence(sv, v = "bill_length_mm", color_var = NULL) &
  geom_smooth(se = FALSE)
# As far as bill length goes, it seems like smaller bills predict Adelie, larger
# for Chinstrap, with Gentoo somewhere in the middle!
# (We need to use `&` instead of `+` because for classification, we get a
# patchwork plot.)



sv_dependence2D(sv, x = "bill_length_mm", y = "body_mass_g")
# What do you make of this plot?






# Some custom plots:
extract_agg_shaps(sv, variables = c("bill_length_mm", "body_mass_g", "sex")) |> 
  # SHAP values are saved in the ".shap" column. For classification, the class
  # is saved in the ".class" column.
  ggplot(aes(bill_length_mm, .shap, color = sex)) + 
  facet_grid(cols = vars(.class),
             rows = vars(cut_number(body_mass_g, 2))) + 
  geom_point() + 
  geom_smooth(se = FALSE, method = "gam")

