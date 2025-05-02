

library(tidymodels)
# library(glmnet)

# Hitters DATA and the PROBLEM ---------------------------------

data("Hitters", package = "ISLR")
ISLR::Hitters
Hitters <- tidyr::drop_na(Hitters, Salary)

# Split:
set.seed(123) 
splits <- initial_split(Hitters, prop = 0.7)
Hitters.train <- training(splits)
Hitters.test <- testing(splits)
# Our data is REALLY SMALL such that splitting the data to train and test might
# leave us with very small datasets.


rec <- recipe(Salary ~ ., data = Hitters.train) |> 
  step_dummy(all_factor_predictors(), one_hot = TRUE) |> 
  # Let's add an interaction here:
  step_interact(~HmRun:starts_with("League")) |> 
  # IMPORTANT! scale the variables as part of the pre-processing
  step_center(all_numeric_predictors()) |> 
  step_scale(all_numeric_predictors())





# Ridge Regression -------------------------

# We will perform ridge regression and the lasso in order to predict Salary on
# the Hitters data.


ridge_spec <- linear_reg(
  mode = "regression", engine = "glmnet", 
  penalty = tune(), mixture = 0
)
# One hyperparameter will be the type of PENALTY: *alpha* argument determines
# the MIXTURE of which models are fit based on the PENALTY - alpha = 0 for a
# ridge regression model, alpha = 1 for lasso model, and 0<alpha<1 for net...we
# will start with ridge.
# Another hyperparameter will be the lambda PENALTY. 
?details_linear_reg_glmnet
translate(ridge_spec)


ridge_wf <- workflow(preprocessor = rec, spec = ridge_spec)



## Tune ------------------------------------

# Let's tune the PENALTY.
ridge_grid <- grid_regular(
  penalty(range = c(-2, 7)),
  
  levels = 20
)

# Using 5-fold CV:
set.seed(12)
cv_5folds <- vfold_cv(Hitters.train, v = 5)


# Tune the model
ridge_tuned <- tune_grid(
  ridge_wf,
  resamples = cv_5folds,
  grid = ridge_grid,
  # Default metrics: rsq, rmse
)

# Let's choose a range
# for lambda values.
autoplot(ridge_tuned) + 
  scale_x_continuous(transform = scales::transform_log(),
                     breaks = scales::breaks_log(n = 10),
                     labels = scales::label_number(big.mark = ",")) + 
  theme(axis.text.x = element_text(angle = 20))

(best_ridge <- select_best(ridge_tuned, metric = "rmse"))


## The final model ------------------------------------------------------

ridge_fit <- ridge_wf |> 
  finalize_workflow(parameters = best_ridge) |> 
  fit(data = Hitters.train)


# We can extract the model's coefficients according to lambda:
ridge_eng <- extract_fit_engine(ridge_fit)

coef(ridge_eng, s = best_ridge$penalty)   
# or other lambdas...
# E.g. for lambda = 0.0000, (this result should be similar to OLS result)
coef(ridge_eng, s = 0)

# Build a function to plot the coefficients with different lambda (s) values
plot_glmnet_coef <- function(mod, s = 0, show_intercept = FALSE) {
  b <- glmnet::coef.glmnet(mod, s = c(s, 0), exact = FALSE) |> 
    as.matrix() |> as.data.frame() |> 
    tibble::rownames_to_column("Coef")
  
  if (isFALSE(show_intercept)) {
    b <- b |> filter(Coef != "(Intercept)")
  }
  
  ggplot2::ggplot(b, ggplot2::aes(Coef, s1)) + 
    ggplot2::geom_hline(yintercept = 0) + 
    ggplot2::geom_point(ggplot2::aes(shape = s1 == 0), fill = "red", size = 2, 
                        show.legend = c(shape = TRUE)) + 
    ggplot2::scale_shape_manual(NULL, 
                                breaks = c(FALSE, TRUE), values = c(16, 24),
                                labels = c("none-0", "0"), 
                                limits = c(FALSE, TRUE)) + 
    ggplot2::scale_x_discrete(guide = ggplot2::guide_axis(angle = 30)) + 
    ggplot2::coord_cartesian(ylim = range(b[,-1])) + 
    ggplot2::labs(y = "Coef", x = NULL) + 
    ggplot2::ggtitle(bquote(lambda==.(s)))
}

# Parameters get smaller as lambda rises:
plot_glmnet_coef(ridge_eng)
plot_glmnet_coef(ridge_eng, s = 1000)
plot_glmnet_coef(ridge_eng, s = 10000)

plot_glmnet_coef(ridge_eng, s = best_ridge$penalty) # some coefs are exactly 0!

# We can also plot the coefficients with the sign of the coefficients using the
# {vip} package:
vip::vip(ridge_eng, method = "model", lambda = 10000, 
         num_features = 100,
         mapping = aes(fill = Sign)) + 
  theme(legend.position = "bottom")


# We can see that the Ridge penalty shrink all coefficients, but doesn't set any
# of them exactly to zero. Ridge regression does not perform variable selection!
#
# This may not be a problem for prediction accuracy, but it can create a
# challenge in model interpretation in settings in which the number of variables
# is large.

# The lasso method overcomes this disadvantage...







# The Lasso -------------------------------------

# As with ridge regression, the lasso shrinks the coefficient estimates towards
# zero. However, Lasso's penalty also force some of the coefficient estimates to
# be exactly equal to zero (when lambda is sufficiently large). Hence, performs
# variable selection.

# We will once again train the model, but we will set alpha = 1.
# Other than that change, we proceed just as we did in fitting a ridge model.


lasso_spec <- linear_reg(
  mode = "regression", engine = "glmnet", 
  penalty = tune(), mixture = 1
)

lasso_wf <- workflow(preprocessor = rec, spec = lasso_spec)


## Tune ---------------------------------------------------------

# This is really the same grid...
lasso_grid <- grid_regular(
  penalty(range = c(-2, 7)),
  
  levels = 20
)

# Tune the model
lasso_tuned <- tune_grid(
  lasso_wf,
  resamples = cv_5folds,
  grid = lasso_grid,
  # Default metrics: rsq, rmse
)

# Let's choose a range
# for lambda values.
autoplot(lasso_tuned) + 
  scale_x_continuous(transform = scales::transform_log(),
                     breaks = scales::breaks_log(n = 10),
                     labels = scales::label_number(big.mark = ",")) + 
  theme(axis.text.x = element_text(angle = 20))

(best_lasso <- select_best(lasso_tuned, metric = "rmse"))



## The final model ------------------------------------------------------

lasso_fit <- lasso_wf |> 
  finalize_workflow(parameters = best_lasso) |> 
  fit(data = Hitters.train)


# We can see that depending on the choice of tuning parameter, more coefficients
# will be EXACTLY equal to zero:
lasso_eng <- extract_fit_engine(lasso_fit)

plot_glmnet_coef(lasso_eng)
plot_glmnet_coef(lasso_eng, s = 10)
plot_glmnet_coef(lasso_eng, s = 100)


# For ridge the lambda was much smaller, but this one is more parsimonious:
# 19 predictors instead of 24!
plot_glmnet_coef(lasso_eng, s = best_lasso$penalty) # some coefs are exactly 0!


# Elastic Net---------------------------------------------------------------

# Elastic Net emerged as a result of critique on lasso, whose variable selection
# can be too dependent on data and thus unstable. The solution is to combine the
# penalties of ridge regression and lasso to get the best of both worlds. alpha
# is the mixing parameter between ridge (alpha=0) and lasso (alpha=1). That is,
# for Elastic Net there are two parameters to tune: lambda and alpha.


enet_spec <- linear_reg(
  mode = "regression", engine = "glmnet", 
  penalty = tune(), mixture = tune()
)

enet_wf <- workflow(preprocessor = rec, spec = enet_spec)




## Tune ---------------------------------------------------------

# This is really the same grid...
enet_grid <- grid_regular(
  penalty(range = c(-2, 7)),
  mixture(),
  
  levels = c(15, 11)
)

# Tune the model
enet_tuned <- tune_grid(
  enet_wf,
  resamples = cv_5folds,
  grid = enet_grid,
  # Default metrics: rsq, rmse
)

# Let's choose a range
# for lambda values.
autoplot(enet_tuned) + 
  scale_x_continuous(transform = scales::transform_log(),
                     breaks = scales::breaks_log(n = 10),
                     labels = scales::label_number(big.mark = ",")) + 
  theme(axis.text.x = element_text(angle = 20))

(best_enet <- select_best(enet_tuned, metric = "rmse"))


## The final model --------------------------------------------


enet_fit <- enet_wf |> 
  finalize_workflow(parameters = best_enet) |> 
  fit(data = Hitters.train)




# Compare performance ---------------------------------------------------------

# EVALUATE the RMSE on the TEST set, associated with this value of lambda:

mset_reg <- metric_set(rsq, mae)

augment(ridge_fit, new_data = Hitters.test) |> mset_reg(Salary, .pred)
augment(lasso_fit, new_data = Hitters.test) |> mset_reg(Salary, .pred)
augment(enet_fit, new_data = Hitters.test) |> mset_reg(Salary, .pred)



# Exercise--------------------------------------------------------------

# Use the "U.S. News and World Report’s College Data" dataset ('College' in
# ISLR). this dataset contains 777 observations of US colleges with the
# following variables:

data("College", package = "ISLR")
head(College)

?College
# Lets predict Grad.Rate (Graduation rate) from these 17 variables.


# 1) Split to train and test. use 0.7 for the train data

# 2) Then, use each of the learned methods to answer this task. That is:
#   i.   Best Subset Selection 
#   ii.  Ridge regression
#   iii. Lasso
#   iv.  Elastic net

# Notes for the last 3 methods, you should use the same lambda values - make
# sure they are broad enough to capture a desired RMSE minima. You can do this
# by plotting RMSE vs lambda and see if there is a "valley". 
# Use 5-folds CV to tune alpha/lambda.

# 3) Compare:
# - Did the method diverged from each other in their performance on test data
#   (look at R2)? Which one preformed best on the test set?
# - Compare the Best Subset Selection, LASSO and Elastic net - did they all
#   "choose" similar predictors?


