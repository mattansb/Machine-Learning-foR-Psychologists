library(tidymodels)
# library(baguette)
# library(randomForest)
# library(xgboost)

mirai::daemons(4) # For parallel processing

# The Boston housing data ----------------------------------------------------

data(Boston, package = "MASS")
?MASS::Boston
glimpse(Boston)


# Split the data:
set.seed(20251201)
splits <- initial_split(Boston, prop = 0.6)
Boston.train <- training(splits)
Boston.test <- testing(splits)

# Split again for CV
Bostin.tune_splits <- vfold_cv(Boston.train, v = 10) # Make 10-folds for tuning
Bostin.comp_splits <- vfold_cv(Boston.test, v = 10) # And 10-folds for comparing

mset_reg <- metric_set(rsq, mae)


# We are again predicting medv (median house value), this time using all 13
# predictors:
rec <- recipe(medv ~ ., data = Boston.train)


# Bagging ---------------------------------------------------------------------
# training the method on B different bootstrapped training data sets. and
# average across all the predictions (for regression) or take the majority
# vote\calculated probability (for classification)

library(baguette)

bag_spec <- bag_tree(
  mode = "regression",
  engine = "rpart",
  # There hyperparameters here similar to those of decision trees, that control
  # the complexity of the tree, but we can keep those at their default
  # (~maximal) values!
  cost_complexity = 0,
  tree_depth = 30,
  min_n = 2
) |>
  # But we do have a special argument - the number of trees. Default is 11 -
  # let's set it higher (only cost is computational!)
  set_args(times = 100)


?details_bag_tree_rpart
translate(bag_spec)
# Actually uses the {baguette} package.

bag_wf <- workflow(preprocessor = rec, spec = bag_spec)

bag_fit <- fit(bag_wf, data = Boston.train)

# We'll use this later for comparisons
bag_resamps <- fit_resamples(
  bag_wf,
  resamples = Bostin.comp_splits,
  metrics = mset_reg
)


## Variable Importance --------------------------------

# We can't look on a specific tree but we can asses variables' importance by
# aggregating across trees. This information is already available inside the
# model:
bag_eng <- extract_fit_engine(bag_fit)

# The VIP shows the mean decrease in node "impurity": the total decrease in node
# impurity that results from splits over that variable, averaged over all trees.
# (In the case of regression trees, the node impurity is measured by the
# training RSS, and for classification trees by the deviance.)
var_imp(bag_eng) |>
  mutate(
    term = forcats::fct_reorder(term, value)
  ) |>
  ggplot(aes(value, term)) +
  geom_col() +
  geom_errorbar(
    aes(xmin = value - std.error, xmax = value + std.error),
    width = 0.2
  )
# The results indicate that across all of the trees considered in the random
# forest, the wealth level of the community (lstat) and the house size (rm) are
# by far the two most important variables.

# Random Forests-------------------------------------------------------------------------------------

# Similar to bagging, with an improvement - when building the trees on the
# bootstrapped training data, each time a split in a tree is considered, a
# random sample of m predictors (mtry parameter) is chosen as split candidates
# from the full set of p predictors-> decorrelating the trees, thereby making
# the average of the resulting trees less variable and hence more reliable.
# Really, bagging is simply a special case of a random forest with m = p.

rf_spec <- rand_forest(
  mode = "regression",
  engine = "randomForest",
  mtry = tune(),
  min_n = 2,
  trees = 100
)

?details_rand_forest_randomForest
translate(rf_spec)


rf_wf <- workflow(preprocessor = rec, spec = rf_spec)

## Tune --------------------------------------

# A reasonable default for mtry:
# - p/3 variables for regression trees,
# - sqrt(p) for classification trees.
# Source:
?randomForest::randomForest
# But we should tune mtry! Let's try the following values (note that we only
# have 13 predictors, so mtry=13 is equivalent to bagging).

# Building the grid ourselves
rf_grid <- expand_grid(
  mtry = c(1, 2, 4, 7, 10, 13)
)

rf_tuner <- tune_grid(
  rf_wf,
  resamples = Bostin.tune_splits,
  grid = rf_grid,
  metrics = mset_reg
)

autoplot(rf_tuner)

select_best(rf_tuner, metric = "mae")
select_by_one_std_err(rf_tuner, mtry, metric = "mae")

# Fit the final model:
rf_fit <- rf_wf |>
  finalize_workflow(
    parameters = select_by_one_std_err(rf_tuner, mtry, metric = "mae")
  ) |>
  fit(data = Boston.train)
rf_fit # mtry = 4


# We'll use this later for comparisons
rf_resamps <- fit_resamples(
  rf_fit,
  resamples = Bostin.comp_splits,
  metrics = mset_reg
)


## Explore the final model ---------------------------------------

rf_eng <- extract_fit_engine(rf_fit)
plot(rf_eng) # See how the error decreased with re-sampling (# of trees)


# Again, we can plot a VIP for the importance of variables *across* all trees.
# This time we will again use the {vip} package:
vip::vip(rf_eng, method = "model", num_features = 13)


# Boosting --------------------------------------------------------------------

# In Bagging \ Random forest, each tree is built on an independent bootstrap
# data. Boosting does not involve bootstrap sampling, and trees are grown
# sequentially:
# - each tree is grown using information from previously grown trees.
# - each tree is fitted using the current residuals, rather than the outcome Y.

# Boosting has three types of tuning parameters:
# 1. Model complexity
# 2. Learning gradient
# 3. Randomness
# https://xgboost.readthedocs.io/en/stable/parameter.html

boost_spec <- boost_tree(
  mode = "regression",
  engine = "xgboost",

  ## Complexity (of each tree)
  tree_depth = 1, # [1, Inf] limits the depth of each tree
  min_n = 2, # [1, Inf] don't split if you get less obs in a node
  loss_reduction = 0, # [0, Inf] node splitting regularization

  ## Gradient
  learn_rate = 0.1, # [0, 1] learning rate
  trees = 100, # [1, Inf]
  # lower learn_rate should come with higher trees

  ## Randomness
  mtry = 4, # Just like mtry in rf
  sample_size = 1 # [0, 1] proportion of random data to use in each tree
)

?details_boost_tree_xgboost
translate(boost_spec)

boost_wf <- workflow(preprocessor = rec, spec = boost_spec)


# Fit the model:
boost_fit <- fit(boost_wf, data = Boston.train)

# We'll use this later for comparisons
boost_resamps <- fit_resamples(
  boost_wf,
  resamples = Bostin.comp_splits,
  metrics = mset_reg
)


## Variable Importance --------------------------------

boost_eng <- extract_fit_engine(boost_fit)
vip::vip(boost_eng, method = "model", num_features = 13)
# We see that lstat and rm are by far the most important variables.

# Comparing ---------------------------------------------------------------

## Compare with resampling --------------------------------

ensemble_metrics <- bind_rows(
  "bagging" = collect_metrics(bag_resamps, summarize = FALSE),
  "rf" = collect_metrics(rf_resamps, summarize = FALSE),
  "boosting" = collect_metrics(boost_resamps, summarize = FALSE),

  .id = "Model"
) |>
  mutate(
    Model = factor(Model, levels = c("bagging", "rf", "boosting"))
  )


ensemble_metrics |>
  group_by(id, .metric) |>
  mutate(
    best_model = case_match(
      .metric,
      "mae" ~ Model[which.min(.estimate)],
      "rsq" ~ Model[which.max(.estimate)]
    )
  ) |>
  ggplot(aes(Model, .estimate, color = Model)) +
  facet_wrap(facets = vars(.metric), scales = "free") +
  stat_summary(size = 1, position = position_nudge(0.1), show.legend = FALSE) +
  geom_point() +
  geom_line(aes(group = id, color = best_model))
# bagging and rf look to be better than boosting in this case.

ensemble_metrics |>
  pivot_wider(names_from = "Model", values_from = ".estimate") |>
  mutate(
    diff = rf - bagging
  ) |>
  summarise(
    mean_diff = mean(diff),
    std_err = sd(diff) / sqrt(n()),
    .lower = mean_diff - 2 * std_err,
    .upper = mean_diff + 2 * std_err,

    .by = .metric
  )
# Nope...

## Test set performance -----------------------------------

Boston.test_rf.pred <- augment(rf_fit, Boston.test)

Boston.test_rf.pred |>
  mset_reg(medv, .pred)

ggplot(Boston.test_rf.pred, aes(.pred, medv)) +
  geom_point() +
  geom_abline() +
  coord_obs_pred()
# That's very good!

# Exercise  ---------------------------------------------------------------

# Use the Hitters dataset.

# Notes: when preparing the data:
# - deal with missing values.
# - split to train and test with p=0.6

# A) Fit *regression* trees to predict Salary from the other variables
# 1. Basic tree - tune the optimal cp with CV.
#    What was the best cp?
# 2. Random Forrest - tune the optimal mtry with CV (include the max possible
#    mrty as one of the candidate values).
#    What was the best mtry?
# 3. Boosting - tune at least one of Complexity and one of the Gradient
#    hyperparameters with CV.
#    What was the best value(s)?

# B) Compare the models:
# 1. Which predictor was most "important" in each method?
# 2. Use CV to see if the two best models differ.
#    - What was the test-set performance of the best model?
