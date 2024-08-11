
library(caret)
library(rsample)
library(yardstick)
library(recipes)

data(Caravan, package = "ISLR")

# Data Splitting
set.seed(1234)
splits <- initial_split(Caravan, prop = 0.7)
Caravan.train <- training(splits)
Caravan.test <- testing(splits)


# The Problem -------------------------------------------------------------

table(Caravan.train$Purchase) |> proportions()
# As we can see, the classes are very unbalanced.
# This means that, technically, we can achieve high accuracy by simply predicting "No".

Caravan.test$pred_NO <- factor("No", levels = c("No", "Yes"))

Caravan.test |> accuracy(truth = Purchase, estimate = pred_NO)

# But as we can see, we essentially have no specificity
table(Truth = Caravan.test$Purchase, 
      Estimate = Caravan.test$pred_NO)

metrics <- metric_set(accuracy, specificity, sensitivity)
Caravan.test |> metrics(truth = Purchase, estimate = pred_NO)

# Training with Imbalance Data --------------------------------------------

rec <- recipe(Purchase ~ ., data = Caravan.train) |> 
  step_scale(all_numeric_predictors())

tc <- trainControl(method = "none", classProbs = TRUE)

tg <- expand.grid(
  k = 10 # [1, N] neighbors
)

knn0 <- train(
  x = rec,
  data = Caravan.train,
  method = "knn", 
  metric = "Accuracy",
  trControl = tc, 
  tuneGrid = tg
)



# Up- and Down-Sampling ---------------------------------------------------

# We can also sample our CV samples such that we artificially achieve class
# balances. See:
# https://topepo.github.io/caret/subsampling-for-class-imbalances.html
#
# The main methods are:
# down-sampling: randomly subset all the classes in the training set so that
#   their class frequencies match the least prevalent class.
# up-sampling: randomly sample (with replacement) the minority class(es) to be
#   the same size as the majority class.
# hybrid methods: techniques such as SMOTE and ROSE down-sample the majority
#   class and synthesize new data points in the minority class. 
#
# We will use up- and down-sampling:
tc_up <- trainControl(method = "none", classProbs = TRUE, sampling = "up")
tc_down <- trainControl(method = "none", classProbs = TRUE, sampling = "down")

knn_up <- train(
  x = rec,
  data = Caravan.train,
  method = "knn", 
  metric = "Accuracy",
  trControl = tc_up, 
  tuneGrid = tg
)

knn_down <- train(
  x = rec,
  data = Caravan.train,
  method = "knn", 
  metric = "Accuracy",
  trControl = tc_down, 
  tuneGrid = tg
)



# Comparing Results -------------------------------------------------------

# Get raw predictions
Caravan.test$pred0 <- predict(knn0, newdata = Caravan.test)
Caravan.test$pred_up <- predict(knn_up, newdata = Caravan.test)
Caravan.test$pred_down <- predict(knn_down, newdata = Caravan.test)

Caravan.test |> metrics(truth = Purchase, estimate = pred_NO)
Caravan.test |> metrics(truth = Purchase, estimate = pred0)
Caravan.test |> metrics(truth = Purchase, estimate = pred_up)
Caravan.test |> metrics(truth = Purchase, estimate = pred_down)

# As we can see, the accuracy (and sensitivity) have dropped, but specificity is
# higher.

# We can also compare ROC AUCs:
# Get probabilistic predictions
Caravan.test$pred0_p <- predict(knn0, newdata = Caravan.test, type = "prob")[["No"]]
Caravan.test$pred_up_p <- predict(knn_up, newdata = Caravan.test, type = "prob")[["No"]]
Caravan.test$pred_down_p <- predict(knn_down, newdata = Caravan.test, type = "prob")[["No"]]

Caravan.test |> roc_auc(truth = Purchase, pred0_p)
Caravan.test |> roc_auc(truth = Purchase, pred_up_p) # About the same?
Caravan.test |> roc_auc(truth = Purchase, pred_down_p) # Better!


library(ggplot2)
update_geom_defaults("path", list(linewidth = 1))

ggplot(mapping = aes(1 - specificity, sensitivity)) + 
  geom_abline(linewidth = 1, linetype = "dashed") + 
  geom_path(aes(color = "(None)"), data = roc_curve(Caravan.test, Purchase, pred0_p)) + 
  geom_path(aes(color = "Up"), data = roc_curve(Caravan.test, Purchase, pred_up_p)) + 
  geom_path(aes(color = "Down"), data = roc_curve(Caravan.test, Purchase, pred_down_p)) + 
  theme_bw() +
  coord_equal() + 
  labs(color = "Sampling")

