### Tutorial 4: Decision Trees ###

library(dplyr)

library(rsample)
library(recipes)
library(caret)
library(yardstick)



# we will also use "MASS" and "randomForest" packages



# Fitting and Evaluating Decision Trees --------------------------------------


## Classification ---------------------------------

### The data --------------------------

# We will use classification trees to analyze the Carseats data set (from ISLR)-
# A simulated data set containing sales of child car seats at 400 different
# stores.
data(Carseats, package = "ISLR")
dim(Carseats)
head(Carseats)

# We want to use Sales as the response. Since it is a continuous variable, we
# will recode it as a binary variable: 'High' is "Yes" if the Sales exceeds 8
# and "No" otherwise.
Carseats <- Carseats |>
  mutate(HighSales = factor(Carseats$Sales <= 8,
                            labels = c("No", "Yes")))

# Base rate:
proportions(table(Carseats$HighSales))




# Let's split the data to train and test sets:
set.seed(42)
splits <- initial_split(Carseats, prop = 0.6)
Carseats.train <- training(splits)
Carseats.test <- testing(splits)



### Fitting a basic classification tree ---------------------------

rec <- recipe(HighSales ~ ., data = Carseats.train) |> 
  # removing Sales from the data as it won't make sense to predict HighSales
  # from Sales
  step_rm(Sales)
# No preprocessing needed here (unless you want to do something...)

tc <- trainControl(method = "cv", number = 5) # using 5-folds CV

tg <- expand.grid(
  cp = 0 # [0, Inf] complexity
)
# cp is the complexity parameter. If set to 0, no pruning is done.




fit.tree <- train(
  x = rec,
  data = Carseats.train,
  method = "rpart", # A simple decision tree is fitted with method = "rpart"
  tuneGrid = tg,
  trControl = tc
) 

fit.tree 
1 - fit.tree$results$Accuracy
# the CV error rate.

## Let's explore the tree:
fit.tree$finalModel

# In each row in the output we see:
# 1. Node index number (where node 1 is the total sample - the ROOT)
# 2. the split criterion 
#    (e.g. ShelveLoc=Good and ShelveLoc=Bad,Medium for the first 2 splits)
# 3. num. of observations under that node 
#    (see how nodes (2) and (3) complete each other)
# 4. the deviance - the number of obs. within the node that deviate from
#    the overall prediction for that node (trees aren't perfect...)
# 5. the overall prediction for the node (Yes/ No)
# 6. in parenthesis - the proportion of observations in that node that take on 
#    values of No (first) or Yes (second) which actually can be calculated using
#    n and deviance.
# * Branches that lead to terminal nodes are indicated using asterisks.

summary(fit.tree) # more detailed results


### Plotting ------------------------


# One of the most attractive properties of trees is that they can be graphically
# displayed:
plot(fit.tree$finalModel) # display the tree structure
text(fit.tree$finalModel, pretty = 0, cex = 0.7) # display the node labels. 
# pretty=0 tells R to include the category names for any qualitative predictors.
# cex is for font size



### Predict and evaluate performance on test set ----------------------------

# Predicting High sales for test data:
Carseats.test$tree_pred <- predict(fit.tree, newdata = Carseats.test)

# We can also get probabilistic predictions:
predict(fit.tree, newdata = Carseats.test, type = "prob") |> head()





# Evaluating test error and other fit indices:
confusionMatrix(Carseats.test$tree_pred, 
                Carseats.test$HighSales, positive="Yes")

# This tree leads to correct predictions for 75% of the test data.
# Which is better than the base rate:
proportions(table(Carseats.test$HighSales))






### Tree Pruning-------------------------------------------------------------

# Next, we consider whether pruning the tree might lead to improved results.

# For finding the best size for the tree using CV change the possible cp
# values for complexity parameters in the tune grid.
tg <- expand.grid(
  cp = seq(0, 0.3, length = 100) # [0, Inf] complexity
)
# check 100 alphas - from 0 to 0.3 


# cp specifies how the cost of a tree is penalized by the number of terminal
# nodes, resulting in a regularized cost for each tress. Small cp results in
# larger trees and potential overfitting (variance), large cp - small trees and
# potential underfitting (bias).

set.seed(42)
fit.tree2 <- train(
  x = rec,
  data = Carseats.train,
  method = "rpart", # A simple decision tree is fitted with method = "rpart"
  tuneGrid = tg,
  trControl = tc
)

# we can see CV-accuracy (or kappa) per cp:
plot(fit.tree2)
# See the drop in accuracy when alpha gets bigger.

# using CV, train() determine the optimal level of tree complexity; cost
# complexity pruning ("weakest link pruning") is used in order to select a
# sequence of best subtrees for consideration, as a function of the tuning
# parameter alpha and here - accuracy:
fit.tree2$bestTune


fit.tree2$finalModel

# In this case, the final model wasn't pruned!
fit.tree2$finalModel$var # all variables where used!

plot(fit.tree2$finalModel) #display the tree structure, only 5 splits
text(fit.tree2$finalModel, pretty = 0, cex = 0.8)

## Evaluate the pruned tree:
Carseats.test$tree_pred2 <- predict(fit.tree2, newdata = Carseats.test)

confusionMatrix(Carseats.test$tree_pred2,
                Carseats.test$HighSales, positive = "Yes")
# Same...



## Regression --------------------------------------------------

# we also have regression problems!

## Boston Data: this data set is included in the MASS library.
data(Boston, package = "MASS")
head(Boston)
dim(Boston)

# The data records medv (median house value) for 506 neighborhoods around
# Boston. We will seek to predict medv using 13 predictors such as:
# rm = average number of rooms per house; 
# age= average age of houses;
# lstat= percent of households with low socioeconomic status.

# The processes fitting and Evaluation of a Regression Tree are essentially the
# same.




# Splitting the data into a train and test sets:
set.seed(42) 
splits <- initial_split(Boston, prop = 0.7)
Boston.train <- training(splits)
Boston.test <- testing(splits)




## Fitting a regression tree on the training data:
rec <- recipe(medv ~ ., data = Boston.train)

tc <- trainControl(method = "cv", number = 5) # using 5-folds CV

tg <- expand.grid(
  cp = seq(0, 0.2, length = 100) # [0, Inf] complexity
)


set.seed(1234)
tree.boston <- train(
  x = rec, 
  data = Boston.train,
  method = "rpart",
  tuneGrid = tg,
  trControl = tc
)

# In the context of a regression tree, cp is chosen based on the CV RMSE:
plot(tree.boston) # RMSE gets bigger when cp rises

tree.boston$bestTune


## The regression tree:
tree.boston$finalModel
plot(tree.boston$finalModel)
text(tree.boston$finalModel,pretty=0,cex=0.55)
# in the terminal nodes we see group means!


## Evaluate the tree performance on test data
Boston.test$pred_tree <- predict(tree.boston, newdata = Boston.test)

rmse(Boston.test, truth = medv, estimate = pred_tree)
rsq(Boston.test, truth = medv, estimate = pred_tree)






# Bagging ---------------------------------------------------------------------

# We apply bagging and random forests to the Boston data.
# The motivation - to reduce the variance when modelling a tree. 

# Bagging:
# training the method on B different bootstrapped training data sets. and
# average across all the predictions (for regression) or take the majority
# vote\calculated probability (for classification)


## Fitting a regression tree to the train data while applying bagging:
# use method= "rf" (as bagging in type of random forest) with maximal number of
# predictors.

tg <- expand.grid(
  mtry = 13 # [1, p] number of random predictors
)
# mtry=13 indicates that all 13 predictors should always be considered:
rec
# In other words, that bagging should be done.
# * this is a fixed hyperparameter for now...




set.seed(1234)
bag.boston <- train(
  x = rec, 
  data = Boston.train,
  method = "rf",
  # ntree = 500,
  tuneGrid = tg,
  trControl = tc
)

bag.boston$finalModel
# see that the default is 500 bootstrapped trees. you can change it by changing
# the ntree (num. of requested trees).


# see how it decreased with re-sampling:
plot(bag.boston$finalModel)


## Variable importance --------------------------------------------------

# We can't look on a specific tree but :(
# but we can asses variables' importance:
bag.vi <- varImp(bag.boston, scale = FALSE)
bag.vi
# varImp() shows the mean decrease in node "impurity": the total decrease in
# node impurity that results from splits over that variable, averaged over all
# trees. (In the case of regression trees, the node impurity is measured by the
# training RSS, and for classification trees by the deviance.)

plot(bag.vi)
# The results indicate that across all of the trees considered in the random
# forest, the wealth level of the community (lstat) and the house size (rm) are
# by far the two most important variables.





## Evaluating the tree performance on test data -------------------------

Boston.test$pred_bag <- predict(bag.boston, newdata = Boston.test)
rmse(Boston.test, truth = medv, estimate = pred_bag)
rsq(Boston.test, truth = medv, estimate = pred_bag)
# This is better than the one obtained using an optimally-pruned single tree!







# Random Forests-------------------------------------------------------------------------------------

# the same principle as for bagging with an improvement - when building the
# trees on the bootstrapped training data, each time a split in a tree is
# considered, a random sample of m predictors is chosen as split candidates from
# the full set of p predictors-> decorrelating the trees, thereby making the
# average of the resulting trees less variable and hence more reliable.

# Recall that bagging is simply a special case of a random forest with m = p.
# Therefore, train() function with method="rf" is used for both analyses.
# Growing a random forest proceeds in exactly the same way, except that we use a smaller
# value of the mtry argument. As a basic default for mtry:
# - p/3 variables for regression trees, 
# - sqrt(p) for classification trees.
# We can also use CV to choose among different mtry. We have 13 predictors, so
# lets try: mtry=4, (closest p/3) and also mtry=2,7,10 for fun, as well as 13
# (i.e. bagging)

tg <- expand.grid(
  mtry = c(2, 4, 7, 10, 13) # [1, p] number of random predictors
)

set.seed(1234)
rf.boston <- train(
  x = rec, 
  data = Boston.train,
  method = "rf",
  # ntree = 500,
  tuneGrid = tg,
  trControl = tc
)

plot(rf.boston)
#The final value used for the model was mtry = 6.
plot(rf.boston$finalModel) 
# 500 trees where fitted for each mtry! that is why it took so long!



# Evaluate
Boston.test$pred_rf <- predict(rf.boston, newdata = Boston.test)
rmse(Boston.test, truth = medv, estimate = pred_rf)
rsq(Boston.test, truth = medv, estimate = pred_rf)
# random forests yielded an improvement over bagging in this case.


rf.vi <- varImp(rf.boston, scale = FALSE)
rf.vi
plot(rf.vi)
# Same conclusions with better fit!





# Boosting --------------------------------------------------------------------
  
# In Bagging \ Random forest, each tree is built on an independent bootstrap
# data. Boosting does not involve bootstrap sampling, and trees are grown
# sequentially:
# - each tree is grown using information from previously grown trees.
# - each tree is fitted using the current residuals, rather than the outcome Y.


# we use method="gbm", but there are many many more methods for boosting:
# https://topepo.github.io/caret/train-models-by-tag.html#boosting

# Boosting has three types of tuning parameters:
# 1. Model complexity
# 2. Learning gradient
# 3. Randomness
  
tg <- expand.grid(
  ## Complexity
  max_depth = 1, # [1, Inf] limits the depth of each tree
  min_child_weight = 5, # [1, Inf] don't split if you get less obs in a node
  gamma = 0, # [0, Inf] node splitting regularization
  
  ## Gradient
  eta = 0.1, # [0, 1] learning rate
  nrounds = 1000, # [1, Inf] number of trees
  # lower eta should come with higher nrounds
  
  ## Randomness
  colsample_bytree = 1, # [0, 1] like mtry in rf
  subsample = 1 # [0, 1] like bagging / rf
)

# A note regarding min_child_weight - What is the best value to use? It depends
# on the data set and whether you are doing classification or regression. Since
# each trees' prediction is taken as the average of the dependent variable of
# all inputs in the terminal node, a value of 1 probably won't work so well for
# regression(!) but may be suitable for classification.
#
# Generally, results are not very sensitive to this parameter, The interaction
# depth, shrinkage and number of trees will all be much more significant!



## Fitting a regression tree to the train data while applying Boosting:
set.seed(1234)
boost.boston <- train(
  x = rec, 
  data = Boston.train,
  method = "xgbTree",
  tuneGrid = tg,
  trControl = tc
)

boost.boston

varImp(boost.boston) |> plot()
# We see that lstat and rm are by far the most important variables. 



## Evaluating the tree performance on test data:
Boston.test$pred_boost <- predict(boost.boston, newdata = Boston.test)
rmse(Boston.test, truth = medv, estimate = pred_boost)
rsq(Boston.test, truth = medv, estimate = pred_boost)
# This is worse than rf, but note we used STUMPS!!
#
# We can improve upon this model by tuning the eta and max_depth parameters.



# Exercise  ---------------------------------------------------------------

# Use the Hitters dataset from the previous tutorial.
# Hitters Dataset: Baseball Data from the 1986 and 1987 seasons
# A data frame with 322 observations of major league players on the following 20
# variables.

# Notes: when preparing the data: 
# - remove cases with NA values.
# - split to train and test with p=0.5

# A) Fit *regression* trees to predict Salary from the other variables
# 1. Basic tree - find the optimal cp with CV.
#    What was the best cp?
# 2. Random Forrest - find the optimal mtry with CV (one value should include
#    the bagging option for this model).
#    What was the best mtry?
# 3. Boosting - tune at least one of shrinkage and interaction.depth with CV.
#    What was the best value(s)?

# B) Compare the models:
# 1. Which has the best CV RMSE?
# 2. Which has the best test RMSE?
# 3. Which predictors were most "important" in each method?
