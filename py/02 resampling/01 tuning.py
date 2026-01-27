import pandas as pd
from plotnine import *

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    make_scorer,
)
from sklearn import set_config

from ISLP import load_data

set_config(display="diagram")


# The data and problem ----------------------------------------------------

# Wage dataset contains information about wage and other characteristics of 3000
# male workers in the Mid-Atlantic region.
Wage = load_data("Wage")
help(Wage)

# Define outcome and predictors
outcome = "wage"
# Remove logwage - we wouldn't want this! (why?)
features = [col for col in Wage.columns if col not in ["wage", "logwage"]]

# Data Splitting (70%):
X_train, X_test, y_train, y_test = train_test_split(
    Wage[features], Wage[outcome], train_size=0.7, random_state=20251201
)

# We'll use KNN - but we will use resampling methods to find K!

# Tuning a KNN model -----------------------------------

## 1) Specify the model -------------------------------------------

# Define the model
# Note: we are setting n_neighbors to a placeholder that will later be tuned.
knn_spec = KNeighborsRegressor()

# Define the preprocessor
# Identify categorical and numerical columns
cat_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# First, dummy code categorical predictors
ct = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ],
    remainder="passthrough",  # Keep numerical columns as-is for now
)

# Then, standardize ALL predictors (including dummy coded ones)
# KNN requires standardization of predictors
preprocessor = Pipeline(
    steps=[
        ("dummy", ct),
        ("scaler", StandardScaler()),
    ]
)

# Create a pipeline
knn_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", knn_spec)]
)

print(knn_pipeline)


## 2) Tune the hyperparameters --------------------------------------------

# To tune a hyperparameter using resampling methods, we need to define:
# 1) A resampling method
# 2) What metrics to use for validation
# 3) How to search for different values of hyperparameters.

### Resampling and Metrics ----------------------------------
# We will use 10-fold CV.

# Define the resampling method (10-fold CV)
cv_folds = KFold(n_splits=10, shuffle=True, random_state=20251202)
# In each "set" we have 1890 obs. for training, and 210 obs. for validation.

# See more methods:
# https://scikit-learn.org/stable/modules/cross_validation.html

# For each fold we will compute the out-of-sample performance using the
# following metrics: R-squared, RMSE, MAE
scoring = {
    "r2": make_scorer(r2_score),
    "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
}


### Tuning method -------------------------------

# In this course we will be tuning by grid search via GridSearchCV function,
# that requires a grid input - predefined candidate values that will be used for
# model fitting and then validation on the validation set(s).

# Define the tuning grid
param_grid = {"regressor__n_neighbors": [5, 10, 50, 200]}

### Model tuning ----------------------------------

# Running many models can be time consuming. We can use parallel processing to
# speed this up with n_jobs parameter.

# Tune the model
knn_tuned = GridSearchCV(
    knn_pipeline,  # the model to re-fit
    param_grid=param_grid,
    cv=cv_folds,
    scoring=scoring,
    refit="rmse",  # which metric to use for selecting the best model
    n_jobs=-1,  # use all available cores
)

knn_tuned.fit(X_train, y_train)


#### View results ---------------------

# Extract the OOS results:
results_df = pd.DataFrame(knn_tuned.cv_results_)
print("\nCross-validation results:")
print(
    results_df[
        [
            "param_regressor__n_neighbors",
            "mean_test_r2",
            "mean_test_rmse",
            "mean_test_mae",
        ]
    ]
)

# We can visualize the tuning results more easily
results_long = results_df.melt(
    id_vars=["param_regressor__n_neighbors"],
    value_vars=["mean_test_r2", "mean_test_rmse", "mean_test_mae"],
    var_name="metric",
    value_name="value",
)

# Adjust RMSE and MAE to be positive for plotting (since they were negative for scoring)
mask = results_long["metric"].isin(["mean_test_rmse", "mean_test_mae"])
results_long.loc[mask, "value"] = -results_long.loc[mask, "value"]

# Rename metrics for better labels
metric_labels = {
    "mean_test_r2": "Rsq",
    "mean_test_rmse": "RMSE",
    "mean_test_mae": "MAE",
}
results_long["metric"] = results_long["metric"].map(metric_labels)

p = (
    ggplot(
        results_long,
        aes(x="param_regressor__n_neighbors", y="value", group="metric"),
    )
    + geom_line()
    + geom_point()
    + facet_wrap("~ metric", scales="free_y", ncol=3)
    + labs(x="K (neighbors)", y="Metric Value", title="Tuning Results")
    + theme_minimal()
    + theme(figure_size=(15, 4))
)
p.draw(show=True)


#### Select hyperparameter values ---------------------
# Select best model
print(f"\nBest parameters: {knn_tuned.best_params_}")
print(f"Best RMSE: {-knn_tuned.best_score_:.3f}")

# Or use the one-SE rule--
# The one-SE rule (selecting the simplest model within one standard error of the
# best model) is not directly implemented in scikit-learn - so we need a little
# helper:)
from one_se_rule import *

low_complexity_k = make_one_se_rule_selector(
    n_fold=10,
    by=["param_regressor__n_neighbors"],
    ascending=False,
    scorer="rmse",
    ret_index=False,
)

params_oneSE = low_complexity_k(results_df)
print(f"\none-SE parameters: {list(params_oneSE.param_regressor__n_neighbors)}")
# See also:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html


## 3) Fit the final model -------------------------------------
# The final model is already fitted with the best parameters during GridSearchCV
knn_final_fit = knn_tuned.best_estimator_


# We can set the new parameter values:
knn_pipeline.set_params(
    regressor__n_neighbors=int(
        params_oneSE.param_regressor__n_neighbors.iloc[0]
    )
)
# And retrain:
knn_pipeline.fit(X_train, y_train)


## 4) Predict and evaluate -------------------------------------------------
# On the test set.

y_pred_test = knn_final_fit.predict(X_test)

print("\n" + "=" * 60)
print("Test set performance:")
print("=" * 60)
print(f"Rsq: {r2_score(y_test, y_pred_test):.3f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred_test):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.3f}")
# Overall, not amazing...

# Exercises ----------------------------------------------------------------------

# - Fit a KNN model - this time include all predictors (except...?)!
#   Use the FULL dataset (without splitting to train/test sets.)
# - Tune the model
#   A. define a grid of K values
#   B. use a metric(s) of your choice
#     https://scikit-learn.org/stable/modules/model_evaluation.html
#   C. Use the following resampling methods:
#     1. LOO-CV
cv_loo = LeaveOneOut()
#     2. 10 repeated 5-fold CV:
cv_repeated = RepeatedKFold(n_splits=5, n_repeats=10)
#   D. Select K using best / one-SE rule.
#     How did the resampling methods differ in their results?
