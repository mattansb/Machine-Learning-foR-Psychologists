# Choosing a threshold ----------------------------------------------------

# This is done *after* model selection, so it is important to use the metric we
# want during training.

# Here is the same example for the 10-fold CV:

library(caret)
library(yardstick)
library(recipes)
library(rsample)

data("Auto", package = "ISLR")
Auto$Economy <- ifelse(Auto$mpg > 30, 1, 0) |> 
  factor(labels = c("Yes", "No"))

set.seed(2345) 
splits <- initial_split(Auto,prop = 0.7)
Auto.train <- training(splits)
Auto.test <- testing(splits)

tc <- trainControl(method = "cv", number = 10,
                   # need these:
                   classProbs = TRUE, 
                   savePredictions = TRUE)
tg <- expand.grid(
  k = c(5:15) # [1, N] neighbors
)

rec <- recipe(Economy ~ horsepower + cylinders + weight, 
              data = Auto.train) |> 
  step_normalize(all_numeric_predictors())

## Find threshold -----------------------------------------------------------

set.seed(9)
# For KNN fitting process, but also- we must set a random seed for, since the
# obs. are sampled into the one of the k folds randomly

knn.fit.10CV <- train(
  x = rec,
  data = Auto.train, 
  method = "knn",
  tuneGrid = tg,
  trControl = tc
)

knn.fit.10CV$bestTune # baset on ACC


resample_stats <- thresholder(knn.fit.10CV,
                              threshold = seq(0.1, 0.9, by = 0.1), 
                              final = FALSE, # look only at the final chosen model
                              statistics = c("F1", "Accuracy", "Sensitivity", "Specificity")) # choose which
resample_stats |> 
  dplyr::arrange(prob_threshold)

# Note that the dataset is not balanced, and still F1 and Accuracy are both best
# at the same threshold of 0.4.
# How about Sens and Spec? Depends on your needs...


## Set threshold to use in training -----------------------------------------

my_metric_fun <- function(data, lev = NULL, model = NULL) {
  truth <- data[["obs"]]
  p <- data[["Yes"]]
  thresh <- 0.4
  pred_raw <- factor(p > thresh, levels = c(TRUE, FALSE), labels = c("Yes", "No"))
  
  c(SENS = sens_vec(truth, pred_raw))
}

tc <- trainControl(method = "cv", number = 10,
                   # need these:
                   classProbs = TRUE, 
                   savePredictions = TRUE, 
                   summaryFunction = my_metric_fun)

knn.fit.10CV2 <- train(
  x = rec,
  data = Auto.train, 
  method = "knn",
  tuneGrid = tg,
  trControl = tc, 
  metric = "SENS"
)

knn.fit.10CV2$bestTune
plot(knn.fit.10CV)
plot(knn.fit.10CV2)
