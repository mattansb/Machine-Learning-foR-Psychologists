# Exercise

Complete the following in R or Python.

## Trees

Use the `Hitters` dataset.

**Notes**: when preparing the data:
- Deal with missing values.
- Split to train (60%) and test (40%)

1. Fit *regression* trees to predict Salary from the other variables
  - _Basic tree_ - tune the optimal cp with CV. (What was the best cp?)
  - _Random Forest_ - tune the optimal mtry with CV. (What was the best mtry?)
  - _Boosting_ - tune at least one of Complexity and one of the Gradient hyperparameters with CV. (What was the best value(s)?)

2. Compare the models:
  - Which predictor was most "important" in each method?
  - Use CV to see if the two best models differ. What was the test-set performance of the best model?
