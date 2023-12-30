

library(recipes)
library(caret)
library(yardstick)

library(lime)
library(kernelshap)
library(shapviz)

# Regression ------------------------------------------------------------------


## The data ---------------------------------------------------------

# Split up the data set
data("Hitters", package = "ISLR")

set.seed(1234)
Hitters.train <- Hitters[-(1:5),] |> tidyr::drop_na(Salary) # all but the first 5 rows
Hitters.test <- Hitters[1:5,] # only the first 5 rows

rec <- recipe(Salary ~ ., data = Hitters.train) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_dummy(all_factor_predictors())

## Tune and Train ---------------------------------

tc <- trainControl(method = "cv", number = 5)

tg <- expand.grid(
  k = c(1, 2, 4, 8, 16) # [1, N]
)

model_reg <- train(
  x = rec, 
  data = Hitters.train,
  method = 'knn', 
  trControl = tc, 
  tuneGrid = tg
)

## Explain Black Box Models ------------------------------------------------


# We've already seen how caret can produce variable importance plots:
# https://topepo.github.io/caret/variable-importance.html

varImp(model_reg, scale = FALSE) |> plot()


# But we will not look at explainer models. There are two types, who do slightly
# different things:
# - LIME (Local Interpretable Model-agnostic Explanations)
# - SHAP (SHapley Additive exPlanations)
#
# Both of these methods are model agnostic - meaning they can work with ANY
# model fitting procedure.

# We will try to EXPLAIN the predictions made...



### LIME --------------------------------
# https://lime.data-imaginist.com/articles/Understanding_lime.html

# The idea behind LIME is that even in complex models, predictions are LOCALLY
# linear. This means that similar observations will get similar predictions that
# are different in understandable ways. 
# By "playing" with different values of features (Xs) and seeing how they change
# the predictions, we can use this idea to see which features drive the
# predictions on a case by case case.
# (Obviously there is a lot more going on here behind the scenes...)
#
# LIME thus tries to answer the Q: 
# "if this X changed, how would Y be affected?"


# The steps:
# 1. Create an explainer from the model and the training data
explainer <- lime(Hitters.train, model = model_reg)

# 2. Use the explainer to explain new cases' predictions:
explanation <- explain(Hitters.test, explainer, n_features = 10)
# n_features controls how many features should be used to explain.

# 3. Explore the explanations:
## - Plot a single explanation:
(p_lime <- plot_features(explanation, cases = rownames(Hitters.test)[2]))

## - Plot a all explanation to look for patterns:
plot_explanations(explanation)


### SHAP ---------------------------------
# https://cran.r-project.org/web/packages/kernelshap/readme/README.html
# https://cran.r-project.org/web/packages/shapviz/vignettes/basic_use.html

# SHAP values try to estimate what each predictor adds, in total, to the
# prediction - accounting for whatever interactions or conditional effects it
# might have. The ideas are based on game theory - really interesting stuff.
# Lots of math.
#
# SHAP thus tries to answer the Q: 
# "if X wasn't in the model at all, how would Y be affected?"

# The steps:
# 1. Train the explainer with the model and the training data (X_bg)
# 2. Use the explainer to explain new cases' predictions:
shaps <- kernelshap(model_reg, X = Hitters.test, bg_X = Hitters.train)

# 3. Explore the explanations:
sv <- shapviz(shaps)

## - Plot a single explanation:
(p_shap <- sv_waterfall(sv, row_id = 2))
# compare to p_lime

## - Plot a all explanation to look for patterns:
sv_importance(sv, max_display = 100)



# Limitations:
# https://proceedings.mlr.press/v119/kumar20e.html




# Classification ------------------------------------------------------------

# It's basically the same, but we need some arguments new in lime() and in
# kernelshap().

## The model ---------------------------------


data("Wage", package = "ISLR")

Wage$maritl3 <- forcats::fct_collapse(
  Wage$maritl, 
  "3. no longer married" = c("3. Widowed", "4. Divorced", "5. Separated")
)

Wage.train <- Wage[-(1:5),] # all but the first 5 rows
Wage.test <- Wage[1:5,] # only the first 5 rows

table(Wage$maritl3)

rec <- recipe(maritl3 ~ ., data = Wage.train) |> 
  step_rm(maritl) |> 
  step_range(all_numeric_predictors()) |> 
  step_dummy(all_factor_predictors())


## Tune and Train ---------------------------------

tc <- trainControl(method = "cv", number = 5)

tg <- expand.grid(
  k = c(1, 2, 4, 8, 16) # [1, N]
)

model_class <- train(
  x = rec, 
  data = Wage.train,
  method = 'knn', 
  trControl = tc, 
  tuneGrid = tg
)




## Explain Black Box Models ------------------------------------------------

varImp(model_class, scale = FALSE) |> plot()

### LIME -----------------

# 1. Create an explainer from the model and the training data
explainer <- lime(Wage.train, model = model_class)

# 2. Use the explainer to explain new cases' predictions:
explanation <- explain(Wage.test, explainer, n_features = 10, 
                       labels = "2. Married")
# n_features controls how many features should be used to explain.
# labels indicates which class we want to explain

# 3. Explore the explanations:
## - Plot a single explanation:
(p_lime <- plot_features(explanation, cases = rownames(Wage.test)[3]))

## - Plot a all explanation to look for patterns:
plot_explanations(explanation)


### SHAP -----------------


# The steps:
# 1. Train the explainer with the model and the training data (X_bg)
#    If there are many training cases (>500), computation time will be huge
#    (especially with many features), so we will look at a random subset.
# 2. Use the explainer to explain new cases' predictions.
shaps <- kernelshap(model_class, X = Wage.test, 
                    bg_X = Wage.train[sample(nrow(Wage.train), 200),], 
                    type = "prob")
# We need to tell the model to predict probabilities


# 3. Explore the explanations:
sv <- shapviz(shaps)

## - Plot a single explanation:
(p_shap <- sv_waterfall(sv, row_id = 3, max_display = 5))
# compare to p_lime

## - Plot a all explanation to look for patterns:
sv_importance(sv, max_display = 100)


# We can see that for different classes, different Xs are important contributes