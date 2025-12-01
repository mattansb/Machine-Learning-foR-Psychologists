library(tidymodels)
# library(rpart)
library(rpart.plot) # For plotting the decision trees


# Classification trees -----------------------------------------------

data("OJ", package = "ISLR")
?ISLR::OJ
glimpse(OJ)
# Which brand of orange juice was purchased?

# Base rate:
table(OJ$Purchase) |> proportions()
# There's a preference for CH!

# Split the data:
set.seed(20251201)
splits <- initial_split(OJ, prop = 0.6)
OJ.train <- training(splits)
OJ.test <- testing(splits)

# Spliting
OJ.tune_splits <- vfold_cv(OJ.train, v = 10) # Make 10-folds for CV
OJ.comp_splits <- vfold_cv(OJ.test, v = 10) # Make 10-folds for CV

# We will use these metric:
# (For some reason, recall of MM is more important than sensitivity/precision.)
f2_meas <- metric_tweak("f2_meas", f_meas, beta = 2)
OJ_metrics <- metric_set(bal_accuracy, f2_meas, roc_auc)


# Our recipe:
rec <- recipe(Purchase ~ ., data = OJ.train) |>
  step_rm(STORE)
# Decision Trees don't require dummy coding or predictor standardization (unless
# you want to do something...)

## Fitting a basic classification tree ---------------------------

OJ.tree_spec <- decision_tree(
  mode = "classification",
  engine = "rpart",
  # Control tree depth
  cost_complexity = 0,
  tree_depth = 30, # Max values is 30 (default)
  min_n = 5 # Default is 2
)
# All the hyperparameters control the complexity and depth of the tree:
# - cost_complexity (cp) is the complexity parameter. If set to 0, no pruning is
#   done.
# - tree_depth is the maximum number of splits ALONG EACH BRANCH.
# - min_n is the minimum number of observations in a node that can be split

?details_decision_tree_rpart
translate(OJ.tree_spec)

OJ.tree_wf <- workflow(preprocessor = rec, spec = OJ.tree_spec)

OJ.tree_fit <- fit(OJ.tree_wf, data = OJ.train)


# One of the most attractive properties of trees is that they can be graphically
# displayed:
extract_fit_engine(OJ.tree_fit) |> rpart.plot(uniform = TRUE)
# The color of the nodes indicates the class prediction at that node, and it's
# saturation indicates how "pure" it is.
# It is quite hard to understand this tree - we need to prune it!

## Tree Pruning-------------------------------------------------------------

# Next, we consider whether pruning the tree might lead to improved results.
pruned.OJ.tree_spec <- decision_tree(
  mode = "classification",
  engine = "rpart",
  cost_complexity = tune(),
  tree_depth = 30,
  min_n = 5
)

pruned.OJ.tree_wf <- workflow(preprocessor = rec, spec = pruned.OJ.tree_spec)


# For finding the best size for the tree using CV change the possible cp
# values for complexity parameters in the tune grid.
pruned.OJ.tree_grid <- grid_regular(
  cost_complexity(range = c(-5, 0)),

  levels = 30
)

pruned.OJ.tree_grid[c(1, 30), ] # We have a wide range of values
# cp specifies how the cost of a tree is penalized by the number of terminal
# nodes, resulting in a regularized cost for each tress. Small cp results in
# larger trees and potential overfitting (variance), large cp - small trees and
# potential underfitting (bias).

pruned.OJ.tree_tuner <- tune_grid(
  pruned.OJ.tree_wf,
  resamples = OJ.tune_splits,
  grid = pruned.OJ.tree_grid,
  metrics = OJ_metrics
)

autoplot(pruned.OJ.tree_tuner) +
  scale_x_continuous(
    transform = scales::transform_log10(),
    breaks = scales::breaks_log(base = 10, n = 10),
    labels = scales::label_number()
  )
# See the drop in performance when cp gets too big.

(pruned.OJ.tree_params <-
  select_by_one_std_err(
    pruned.OJ.tree_tuner,
    desc(cost_complexity),
    # smaller values of cp lead to more complex models
    metric = "roc_auc"
  ))


# Fit the final model:
pruned.OJ.tree_fit <- pruned.OJ.tree_wf |>
  finalize_workflow(parameters = pruned.OJ.tree_params) |>
  fit(data = OJ.train)


## Let's explore the tree:

pruned.OJ.tree_eng <- extract_fit_engine(pruned.OJ.tree_fit)
pruned.OJ.tree_eng
# In each row in the output we see:
# 1. Node index number (where node 1 is the total sample - the ROOT)
# 2. the split criterion
# 3. num. of observations under that node
#    (see how nodes (2) and (3) complete each other)
# 4. the deviance - the number of obs. within the node that deviate from
#    the overall prediction for that node (trees aren't perfect...)
# 5. the overall prediction for the node (MM/ CH)
# 6. in parenthesis - the proportion of observations in that node that take on
#    values of No (first) or Yes (second) which actually can be calculated using
#    n and deviance.
# * Branches that lead to terminal nodes are indicated using asterisks.

summary(pruned.OJ.tree_eng) # more detailed results


# We can now plot a MUCH smaller tree:
rpart.plot(
  pruned.OJ.tree_eng,
  type = 2,
  extra = 104,
  # Should the length of the branches NOT be spaced proportionally to
  # the fit improvement?
  uniform = TRUE
)
# In each node we get:
# - Predicted class (also corresponds to the color)
# - Class proportions in the node (also corresponds to the saturation of color)
# - % of sample in that node/branch

# (When uniform = FALSE) we also get a visual indication of the value of each
# split in improving the model's fit.
# We can summarize this information by predictor, using an variable importance
# plot (VIP):
vip::vip(pruned.OJ.tree_eng, method = "model", num_features = 20)
# A variable's importance is the sum of the goodness of split measures for each
# split for which it was the primary variable + (goodness * agreement) for all
# splits in which it was a surrogate (=when it was correlated with a different
# split on another variable).

# We will discuss a general framework for estimating variables' importance later
# in the semester.

## Compare ---------------------------------

OJ.test_predictions.tree <- augment(OJ.tree_fit, new_data = OJ.test)
OJ.test_predictions.pruned.tree <- augment(
  pruned.OJ.tree_fit,
  new_data = OJ.test
)

OJ.test_predictions.tree |> conf_mat(Purchase, .pred_class)
OJ.test_predictions.pruned.tree |> conf_mat(Purchase, .pred_class)
# Models seem to give different predictions...

OJ.test_predictions.tree |>
  OJ_metrics(Purchase, estimate = .pred_class, .pred_CH)
OJ.test_predictions.pruned.tree |>
  OJ_metrics(Purchase, estimate = .pred_class, .pred_CH)


bind_rows(
  "tree" = OJ.test_predictions.tree,
  "pruned" = OJ.test_predictions.pruned.tree,
  .id = "Model"
) |>
  group_by(Model) |>
  roc_curve(Purchase, .pred_CH) |>
  autoplot()
# It seems the pruned tree is better on all metrics.

# Let's use CV to compare the trees:
OJ.tree_resamps <- fit_resamples(
  OJ.tree_wf,
  resamples = OJ.comp_splits,
  metrics = OJ_metrics
)
pruned.OJ.tree_resamps <- fit_resamples(
  pruned.OJ.tree_fit,
  resamples = OJ.comp_splits,
  metrics = OJ_metrics
)


OJ_resamps_metrics <- bind_rows(
  "tree" = collect_metrics(OJ.tree_resamps, summarize = FALSE),
  "pruned" = collect_metrics(pruned.OJ.tree_resamps, summarize = FALSE),

  .id = "Model"
) |>
  group_by(id, .metric) |>
  mutate(
    best_is = Model[which.max(.estimate)]
  ) |>
  ungroup()


ggplot(OJ_resamps_metrics, aes(Model, .estimate, color = Model)) +
  facet_wrap(~.metric, scales = "free") +
  geom_line(aes(group = id, color = best_is)) +
  geom_point() +
  stat_summary(
    aes(fill = Model),
    geom = "point",
    size = 3,
    shape = 21,
    color = "black"
  )
# We can see that the pruned tree was better across most folds/metrics!

# Regression trees --------------------------------------------------

# we also have regression problems!

data(Boston, package = "MASS")
?MASS::Boston
glimpse(Boston)


# We won't be splitting the data into test/train for this example.

# Split for CV (tune)
Boston.folds <- vfold_cv(Boston, v = 10) # Make 10-folds for CV


# The data records medv (median house value) for 506 neighborhoods around
# Boston. We will seek to predict medv using 13 predictors such as:
# rm = average number of rooms per house;
# age = average age of houses;
# lstat = percent of households with low socioeconomic status.

rec <- recipe(medv ~ ., data = Boston)


## Tune ----------------------------------------------------------------

# The processes fitting and Evaluation of a Regression Tree are essentially the
# same.

Boston.tree_spec <- decision_tree(
  mode = "regression",
  engine = "rpart",
  cost_complexity = tune(),
  tree_depth = 30,
  min_n = 5
)


Boston.tree_wf <- workflow(preprocessor = rec, spec = Boston.tree_spec)


Boston.tree_grid <- grid_regular(
  cost_complexity(range = c(-5, 0)),

  levels = 20
)

Boston.tree_tuned <- tune_grid(
  Boston.tree_wf,
  resamples = Boston.folds,
  grid = Boston.tree_grid
)

autoplot(Boston.tree_tuned) +
  scale_x_continuous(
    transform = scales::transform_log10(),
    breaks = scales::breaks_log(base = 10),
    labels = scales::label_number()
  )


# Fit the final model:
Boston.tree_fit <- Boston.tree_wf |>
  finalize_workflow(
    parameters = select_best(Boston.tree_tuned, metric = "rmse")
  ) |>
  fit(data = Boston)


## Explore the pruned tree ---------------------------------

Boston.tree_eng <- extract_fit_engine(Boston.tree_fit)
Boston.tree_eng
# In each row in the output we see:
# 1. Node index number (where node 1 is the total sample - the ROOT)
# 2. the split criterion
# 3. num. of observations under that node
#    (see how nodes (2) and (3) complete each other)
# 4. the deviance (variance) within that node (trees aren't perfect...)
# 5. the overall prediction for the node (Yes/ No)
# * Branches that lead to terminal nodes are indicated using asterisks.

summary(Boston.tree_eng) # more detailed results


# We can now plot a MUCH smaller tree:
rpart.plot(
  Boston.tree_eng,
  type = 2,
  extra = 101,
  # Should the length of the branches NOT be spaced proportionally to
  # the fit improvement?
  uniform = TRUE
)
# In each node we get:
# - Predicted class (also corresponds to the color)
# - % of sample in that node/branch

# Again, (when uniform = FALSE) we also get a visual indication of the value of
# each split in improving the model's fit.
# We can summarize this information by predictor, using an variable importance
# plot (VIP):
vip::vip(Boston.tree_eng, method = "model")
# The most important splits occur along the "rm" predictor.
# Note that not all 13 predictors appear.
