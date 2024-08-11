
library(caret) 
library(rsample)
library(recipes)
library(yardstick)

library(ggplot2)

# https://rsample.tidymodels.org/articles/Common_Patterns.html

# We will use the sleepstudy dataset from the lme4 package
data("sleepstudy", package = "lme4")
head(sleepstudy)

# This data has *REPEATED MEASURES* from 18 subjects
nlevels(sleepstudy$Subject)

# This means we need to take care when splitting the data - both for the initial
# split and any CV/bootstrap.

# The right way --------------------------------------------------------------

## Initial split ----------------------------------------------
# Instead of initial_split() use group_initial_split():

set.seed(1)
init <- group_initial_split(sleepstudy, group = Subject, prop = 0.7)
sleepstudy.train <- training(init)
sleepstudy.test <- testing(init)

# Non of the subjects in the train set appear in the test set!
any(sleepstudy.train$Subject %in% sleepstudy.test$Subject)


## Define CV folds -------------------------------------------------------
# We will not rely on {caret} to do this for us. Instead we will build our own
# sets using {rsample}.


set.seed(44)
gfolds_5 <- group_vfold_cv(sleepstudy.train, group = Subject, v = 5)
gfolds_5.caret <- rsample2caret(gfolds_5)
names(gfolds_5.caret) # a list of lists
# Each element in `index` (row indices of a training set) has a corresponding
# element in `indexOut` (row indices of a vailidation set) - in this case, of
# DIFFERENT subjects.


## Train: A simple linear regression ------------------------------------------

rec <- recipe(Reaction ~ Days, 
              data = sleepstudy.train)


# We will now use the 5 folds defined above:
tc <- trainControl(index = gfolds_5.caret$index,
                   indexOut = gfolds_5.caret$indexOut)


# We will use train().
fit_lin <- train(
  x = rec,
  data = sleepstudy.train,
  method = "lm",
  trControl = tc
)

fit_lin$results # out-of-sample performance on *independent* validation sets

## Test -----------------------------------------------------------------------

sleepstudy.test$pred <- predict(fit_lin, newdata = sleepstudy.test)

ggplot(sleepstudy.test, aes(pred, Reaction, color = Subject)) +
  geom_abline() + 
  geom_point() + 
  coord_equal(xlim = c(200, 450), ylim = c(200, 450))

rsq(sleepstudy.test, truth = Reaction, estimate = pred)
# Not bad!

# The WRONG way --------------------------------------------------------------

# Let's do the whole thing again, ignoring the dependancy.
# Will we see overfitting?

## Initial split ----------------------------------------------
# Using initial_split():

set.seed(2)
init2 <- initial_split(sleepstudy, prop = 0.7)
sleepstudy.train2 <- training(init2)
sleepstudy.test2 <- testing(init2)

# Some of the subjects in the train set appear in the test set!
any(sleepstudy.train2$Subject %in% sleepstudy.test2$Subject)


## Define CV folds -------------------------------------------------------
# For completeness, we will again use sample do define standard 5-fold CV:


set.seed(44)
folds_5 <- vfold_cv(sleepstudy.train2, v = 5)
folds_5.caret <- rsample2caret(folds_5)
names(folds_5.caret) # a list of lists
# Again, each element in `index` (row indices of a training set) has a
# corresponding element in `indexOut` (row indices of a vailidation set).


## Train: A simple linear regression ------------------------------------------

rec2 <- recipe(Reaction ~ Days, 
              data = sleepstudy.train2)


# We will now use the 5 folds defined above:
tc2 <- trainControl(index = folds_5.caret$index,
                    indexOut = folds_5.caret$indexOut)


# We will use train().
fit_lin2 <- train(
  x = rec2,
  data = sleepstudy.train2,
  method = "lm",
  trControl = tc2
)

## Compare -----------------------------------------------------------

## CV results:
rbind(
  "Accounting for dep." = fit_lin$results,
  "Ignoring dep." = fit_lin2$results
)
# Looks at the difference in RMSE and Rsq! We are over estimating the
# out-of-sample performance by ignoring the dependency!

sleepstudy.test2$pred <- predict(fit_lin2, newdata = sleepstudy.test2)

rsq(sleepstudy.test, truth = Reaction, estimate = pred)
rsq(sleepstudy.test2, truth = Reaction, estimate = pred)

rmse(sleepstudy.test, truth = Reaction, estimate = pred)
rmse(sleepstudy.test2, truth = Reaction, estimate = pred)
# Again - we are over estimating our out of sample performance :(

