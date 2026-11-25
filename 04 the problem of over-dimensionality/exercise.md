# Exercise

Complete the following in R or Python.

## Regularization

```R
data("attrition", package = "modeldata")
?modeldata::attrition
```

The attrition dataset dataset contains information about employees in a company and whether they left the company (_Attrition: Yes/No_).
The goal is to predict Attrition using the 30 other variables in the dataset.
(Note that regularization methods can be used for datasets with many more variables than this one).


1. Make sure the level of `Attrition` are coded to correctly identify the "target" level (`"Yes"`).

2. Split to train and test. use 0.7 for the train data

3. Set up an appropriate preprocessing pipeline.

4. Then, use each of the learned methods to answer this task. That is:
  i.   Ridge regression
  ii.  Lasso
  iii. Elastic net

  - Define a broad enough range of lambda values for all three methods.
    You can do this by plotting a metric vs lambda and see if there is a "valley" / "hill".
  - Use appropriate metrics to evaluate the performance of the models
  - Use 10-folds CV to tune alpha/lambda.

5. Compare:
  - Did the methods diverged from each other in their performance on test data?
    Which one preformed best on the test set?
  - Did LASSO and Elastic net "choose" similar predictors?
