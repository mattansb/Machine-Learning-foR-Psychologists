
# library(ISLR)

library(caret) 
library(rsample)
library(recipes)
library(yardstick)

library(tidyverse)



# Data, split and recipe -----------------------------------------------------------

data(Smarket, package = "ISLR")

set.seed(1234)
splits <- initial_split(Smarket, prop = 0.7)
Smarket.train <- training(splits)
Smarket.test <- testing(splits)

# Fit some models --------------------------------------------------

## Setting up custom CV folds.... ------------------------------------------

# Since we want all the models to go through the same *exact* training and
# validation sets, we will not rely on {caret} to do this for us. Instead we
# will build our own sets using {rsample}.

set.seed(44)
folds_10 <- vfold_cv(Smarket.train, v = 10)
folds_10.caret <- rsample2caret(folds_10)
names(folds_10.caret)


## Train 1: Logistic regression --------------------------------------------

rec1 <- recipe(Direction ~ ., 
               data = Smarket.train)


# We will now use the 10 folds defined above to asses model performance:
tc1 <- trainControl(index = folds_10.caret$index,
                    indexOut = folds_10.caret$indexOut)


# We will use train().
fit_logistic1 <- train(
  x = rec1,
  data = Smarket.train,
  method = "glm", family = binomial("logit"),
  trControl = tc1
)


## Train 2: Logistic regression --------------------------------------------

# This time, we will use different predictors:
rec2 <- recipe(Direction ~ Lag1 + Lag2 + Lag3, 
               data = Smarket.train)


# We will use train().
fit_logistic2 <- train(
  x = rec2,
  data = Smarket.train,
  method = "glm", family = binomial("logit"),
  trControl = tc1 # # Same as before
)

## Train 3: KNN --------------------------------------------

# This time, we will use different predictors:
rec3 <- recipe(Direction ~ ., 
               data = Smarket.train) |> 
  step_normalize(all_predictors())

tg3 <- expand.grid(
  k = c(20, 50, 100, 200, 300) # [1, N] neighbors 
)

# We are using CV both to asses model performance *and* to select k!
tc3 <- trainControl(index = folds_10.caret$index,
                    indexOut = folds_10.caret$indexOut, 
                    selectionFunction = "best")

set.seed(123)
# We will use train().
fit_KNN <- train(
  x = rec3,
  data = Smarket.train,
  method = "knn",
  tuneGrid = tg3, # our K's for KNN
  trControl = tc3
)

plot(fit_KNN)
fit_KNN$results



# Compare models based on CV ------------------------------------------------

# The out-of-sample performance for each fold.
# Remember we use the same folds for each training!
cv_data <- list(
  logistic1 = fit_logistic1$resample,
  logistic2 = fit_logistic2$resample,
  KNN = fit_KNN$resample
) |> 
  bind_rows(.id = "Model")

# Summary across folds
cv_summary <- list(
  logistic1 = fit_logistic1$results,
  logistic2 = fit_logistic2$results,
  KNN = fit_KNN$results[2,]
) |> 
  bind_rows(.id = "Model") |> 
  # Compute SE
  mutate(
    AccuracySE = AccuracySD / sqrt(5)
  )

## Plot -----------------

ggplot(cv_data, aes(Model, Accuracy)) + 
  geom_hline(yintercept = 0.5, linetype = "dashed") + 
  geom_line(aes(group = Resample)) + 
  geom_pointrange(aes(ymin = Accuracy - AccuracySE,
                      ymax = Accuracy + AccuracySE),
                  data = cv_summary,
                  color = "red") +
  coord_cartesian(ylim = c(NA, 1)) + 
  theme_bw()

## Contrast ----------------

# We need the correlations between the model's fold-wise performance.
cv_data_wide <- cv_data |> 
  select(-Kappa) |> 
  pivot_wider(names_from = Model, 
              values_from = Accuracy, 
              id_cols = Resample) |> 
  column_to_rownames(var = "Resample")
cv_cor <- cor(cv_data_wide)
  
# Compare (almost like a paired t-test...)
d_log1.vs.log2 <- cv_summary$Accuracy[1] - cv_summary$Accuracy[2]
se_log1.vs.log2 <- sqrt(cv_summary$AccuracySE[1]^2 + cv_summary$AccuracySE[2]^2 - 2*cv_cor[1,2]*cv_summary$AccuracySE[1]^2 + cv_summary$AccuracySE[2]^2)
c(d_log1.vs.log2 - se_log1.vs.log2, d_log1.vs.log2 + se_log1.vs.log2)





# Performance of selected model on test-set -----------------------------------

Smarket.test$Direction_prob <- predict(fit_logistic1, newdata = Smarket.test, type = "prob")[,"Up"]
Smarket.test$Direction_class <- predict(fit_logistic1, newdata = Smarket.test, type = "raw")

Smarket.test |> 
  roc_curve(Direction, Direction_prob, 
            event_level = "second") |> 
  autoplot()

Smarket.test |> roc_auc(Direction, Direction_prob, event_level = "second")
Smarket.test |> f_meas(Direction, Direction_class, event_level = "second", beta = 1)

# etc.....

