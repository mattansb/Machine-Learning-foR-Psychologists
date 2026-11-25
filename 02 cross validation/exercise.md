# Exercise

Complete the following in R or Python.

## Cross Validation

- Fit a KNN model
  - Include all predictors (except...?)!
  - Use the **full** dataset (without splitting to train/test sets.)
- Tune the model:

  1. Define a grid of K values

  2. Use a metric(s) of your choice;
    See relevant help pages for
    [`{yardstick}`](https://yardstick.tidymodels.org/reference/index.html) or
    [`scikit-learn`](https://scikit-learn.org/stable/modules/model_evaluation.html).

  3. Use the following resampling methods:

    ```R
    # 1. With 50 bootstrap samples
    # (Note validation set is not always of the same size!)
    bootstrap_samps <- bootstraps(Wage, times = 50)
    # 2. 10 repeated 5-fold CV:
    cv_repeated_folds <- vfold_cv(Wage, v = 5, repeats = 10)
    ```

    ```python
    # 1. LOO-CV
    cv_loo = LeaveOneOut()
    # 2. 10 repeated 5-fold CV:
    cv_repeated = RepeatedKFold(n_splits=5, n_repeats=10)
    ```

  4. Select K using best / one-SE rule.
    - How did the resampling methods differ in their results?
