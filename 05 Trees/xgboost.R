# https://www.kaggle.com/code/pelkoja/visual-xgboost-tuning-with-caret
# http://ml-course.kazsakamoto.com/Labs/hyperparameterTuning.html

tg <- expand.grid(
  nrounds = 1000, # number of trees
  max_depth = 1, # limits the depth of each tree
  eta = 0.1, # learning rate
  min_child_weight = 5, # don't split if you get less obs in a node
  
  gamma = 0,
  colsample_bytree = 1, 
  subsample = 1
)

set.seed(1234)
xgboost.boston <- train(
  x = rec, 
  data = Boston.train,
  method = "xgbTree",
  tuneGrid = tg,
  trControl = tc
)

xgboost.boston

varImp(xgboost.boston) |> plot()
