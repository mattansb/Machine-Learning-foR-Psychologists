import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    ShuffleSplit,
    RepeatedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    roc_auc_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    RocCurveDisplay,
)

from ISLP import load_data

# The data and problem ----------------------------------------------------

# Smarket dataset contains daily percentage returns for the S&P 500 stock index
# between 2001 and 2005 (1,250 days).
Smarket = load_data("Smarket")
print(Smarket.info())
# For each date, the following vars were recorded:
# - Lag1--Lag5 - percentage returns for each of the five previous trading days.
# - Volume - the number of shares traded on the previous day(in billions).
# - Today - the percentage return on the date in question.
# - Direction - whether the market was Up or Down on this date.

# # To make life easier, we will relevel the factor so that the positive class is
# # FIRST (which is the default behavior in {yardstick}).
# print(Smarket.Direction.cat.categories)


# Assume the following classification task on the Smarket data:
# predict Direction (Up/Down) using the features Lag1 and Lag2.
# If we are not sure how Direction is coded we can use levels():
print(Smarket.Direction.cat.categories)

print(Smarket.Direction.value_counts())
# The base rate probability:
print(Smarket.Direction.value_counts(normalize=True))


# Data Splitting (70%):
outcome = "Direction"
features = Smarket.columns.difference(
    [outcome, "Volume", "Today"]
)  # Keep only Lag predictors

X_train, X_test, y_train, y_test = train_test_split(
    Smarket[features], Smarket[outcome], train_size=0.7, random_state=1234
)

# We'll use KNN - but we will use resampling methods to find K!

# Tuning a KNN model -----------------------------------

## 1) Specify the model -------------------------------------------

knn_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),  # Standardize features
        ("knn", KNeighborsClassifier()),  # KNN classifier
    ]
)

## 2) Tune the hyperparameters --------------------------------------------

# To tune a hyperparameters using resampling methods, we need to define:
# 1) A resmapling method
# 2) What metrics to use for validation
# 3) How to search for different values of hyperparameters.


### Resampling and Metrics ----------------------------------
# We will use 10-fold CV.

# Define the resampling method (10-fold CV)
splits = KFold(n_splits=10)

# See more methods:
# https://scikit-learn.org/stable/api/sklearn.model_selection.html#splitters


# For each fold we will compute the out-of-sample performance using the
# following scorers:
scoring_clf = {
    "accuracy": make_scorer(accuracy_score),
    "prec": make_scorer(precision_score, pos_label="Up"),
    "f1": make_scorer(f1_score, pos_label="Up"),
    "auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "brier": make_scorer(
        brier_score_loss,
        pos_label="Up",
        response_method="predict_proba",
        greater_is_better=False,
    ),
}
# Not we have to tell make_scorer() if it should use a prediction method
# other than predict() or if should expect smaller rathar than larger
# values, or if any parameters need to be passed to the scoring function.

### Tuning method -------------------------------

# In this course we will be tuning by grid search - via the
help(GridSearchCV)
# object, that requires a grid input - predefined candidate values that will
# be used for model fitting and then validation on the validation set(s).
#
# See more options here:
# https://scikit-learn.org/stable/api/sklearn.model_selection.html#hyper-parameter-optimizers

# Define the tuning grid
knn_grid = {"knn__n_neighbors": np.array([5, 10, 50, 200])}
# Note the name is "{name of estimator in pipe}__{name of parameter}"


### Model tuning ----------------------------------

grid_search = GridSearchCV(
    estimator=knn_pipe,
    param_grid=knn_grid,
    cv=splits,
    scoring=scoring_clf,
    refit="auc",
    verbose=1,  # Show progress
    n_jobs=-1,  # Use all available CPU cores
)

grid_search.fit(X_train, y_train)

#### View results ---------------------
cv_res = pd.DataFrame(grid_search.cv_results_)

# Plot the results:
cv_res.plot(x="param_knn__n_neighbors", y="mean_test_auc", kind="line")
cv_res.plot(x="param_knn__n_neighbors", y="mean_test_accuracy", kind="line")

# Let's take AUC as an example
auc_cv_res = cv_res[["params", "mean_test_auc", "std_test_auc"]].assign(
    # convert std to SE
    se_test_auc=lambda df: df.std_test_auc / np.sqrt(10)
)
print(auc_cv_res)

#### Select hyperparameter values ---------------------

# Select best model
best_k = (
    auc_cv_res.sort_values(by="mean_test_auc", ascending=False)
    .reset_index()
    .loc[0, "params"]
)
print(best_k)

# This is also what is doe automatically when setting refit="auc" above:
print(grid_search.best_params_)

# The best model is also automarically refit to the full training set and
# returned here:
knn_best = grid_search.best_estimator_


# Or use the one-SE rule
# (for this we need a little helper:)
from one_se_rule import *

low_complexity_k = make_one_se_rule_selector(
    n_fold=10,
    by=["param_knn__n_neighbors"],
    ascending=False,
    scorer="auc",
    ret_index=False,
)

params_oneSE = low_complexity_k(cv_res)
# See also:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html

print(params_oneSE)

# We can set the new parameter values:
knn_pipe.set_params(
    knn__n_neighbors=int(params_oneSE.param_knn__n_neighbors.iloc[0])
)
# And retrain:
knn_pipe.fit(X_train, y_train)


## 3) Predict and evaluate -------------------------------------------------

y_pred = knn_best.predict(X_test)
y_pred_propUp = knn_best.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))


acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_propUp)

print(f"Acuracy = {acc:.3f}")
print(f"AUC = {auc:.3f}")
# Overall, not amazing...


# Since this is a probabilistic model, we can also look at the ROC curve and
# AUC:
plt.figure()
RocCurveDisplay.from_predictions(
    y_test, y_pred_propUp, pos_label="Up", plot_chance_level=True
)
plt.show()


# Exercises ----------------------------------------------------------------------

# - Fit a KNN model - this time include all predictors!
#   Use the FULL dataset (without splitting to train/test sets.)
# - Tune the model
#   A. define a grid of K values
#   B. use a metric(s) of your choice
#     https://scikit-learn.org/stable/api/sklearn.metrics.html#classification-metrics
#   C. Use the following resampling methods:
#     1. With 50 Random permutation
ShuffleSplit(n_splits=50)
#     2. 10 repeated 5-fold CV:
RepeatedKFold(n_repeats=10, n_splits=5)
#   D. Select K using best / one-SE rule.
#     How did the resampling methods differ in their results?
