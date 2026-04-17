import pandas as pd
import numpy as np

from plotnine import *

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    KFold,
    GroupKFold,
    cross_validate,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    make_scorer,
)

# We will use the hotel_rates dataset from {tidytuesday}:
# https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11
hotel_rates = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv"
)
hotel_rates.info()

# This data has *REPEATED MEASURES* from 117 countries
hotel_rates[["country", "assigned_room_type"]] = hotel_rates[
    ["country", "assigned_room_type"]
].astype("category")
print(len(hotel_rates["country"].cat.categories))

# This means we need to take care when splitting the data - both for the initial
# split and any CV/bootstrap.

# For this demo, we'll use data from the first 15
hotel_rates = hotel_rates.loc[hotel_rates["hotel"] == "Resort Hotel", :]
hotel_rates = hotel_rates.loc[
    hotel_rates["country"].isin(hotel_rates["country"].cat.categories[0:15]), :
]


# Initial split --------------------------------------------------------------

# PROBLEM: Standard train/test split ignores grouping structure
# Using a standard split:
ht_train_standard, ht_test_standard = train_test_split(
    hotel_rates, train_size=(2 / 3), random_state=111
)

# Check if test countries appear in training set
has_overlap = any(
    ht_test_standard["country"].isin(ht_train_standard["country"])
)
print(f"Countries overlap between train and test: {has_overlap}")
# This is a problem! We have data leakage.

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py


# SOLUTION: Use grouped split to respect data structure
# GroupShuffleSplit ensures entire groups (countries) are kept together
splitter_grouped = GroupShuffleSplit(
    n_splits=1, train_size=(2 / 3), random_state=111
)

# Get indices for train/test split that respect country grouping
train_idx, test_idx = next(
    splitter_grouped.split(hotel_rates, groups=hotel_rates["country"])
)

# Create train and test sets
ht_train = hotel_rates.iloc[train_idx].copy()
ht_test = hotel_rates.iloc[test_idx].copy()

# Verify no country overlap
has_overlap_grouped = any(ht_test["country"].isin(ht_train["country"]))
print(f"Countries overlap with grouped split: {has_overlap_grouped}")
# Perfect! No countries in the test set appear in the training set.


# Assessing OOS performance using CV -----------------------------------------
# One way to assess the out-of-sample (oos) performance of a model is to use
# cross-validation. This is done by splitting the data into k-folds, training
# the model on k-1 of the folds and validating on the remaining fold. By doing
# this we can get the mean oos performance, but also the between fold
# variability (as realized in the standard error).

# We will be using a linear regression here, looking at R2 and RMSE, but the
# same principles apply to other models and metrics.

outcome = "adr"  # Average Daily Rate
predictors = [
    "adults",
    "children",
    "babies",
    "total_of_special_requests",
    "assigned_room_type",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "dummy",
            OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist"),
            ["assigned_room_type"],
        ),
        (
            "pass",
            "passthrough",
            [
                "adults",
                "children",
                "babies",
                "total_of_special_requests",
            ],
        ),
    ]
)

linreg_pipe = Pipeline(
    steps=[
        ("preproc", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

scorers_reg = {
    "rsq": make_scorer(r2_score),
    "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
}

## Standard CV (IGNORING grouping structure) -----------------------------

# Standard K-Fold CV - doesn't consider grouping
folds_standard = KFold(n_splits=5, shuffle=True, random_state=111)

# Run cross-validation
linreg_cv_standard = cross_validate(
    estimator=linreg_pipe,
    X=ht_train[predictors],
    y=ht_train[outcome],
    cv=folds_standard,
    scoring=scorers_reg,
    return_train_score=False,
    n_jobs=-1,
)
# PROBLEM: Data from each country are scattered across folds
# This can lead to overoptimistic performance estimates

## Grouped CV (RESPECTING grouping structure) -----------------------------

# GroupKFold CV - keeps entire groups together
folds_grouped = GroupKFold(n_splits=5)

# Run cross-validation with grouped folds
linreg_cv_grouped = cross_validate(
    estimator=linreg_pipe,
    X=ht_train[predictors],
    y=ht_train[outcome],
    cv=folds_grouped,
    groups=ht_train["country"],  # Specify the grouping variable
    scoring=scorers_reg,
    return_train_score=False,
    n_jobs=-1,
)
# SOLUTION: Each country appears in only one fold
# This gives more realistic performance estimates


## Compare standard vs grouped CV -------------------------------


def std_err(x):
    """Calculate standard error of the mean"""
    return x.std() / np.sqrt(len(x))


# Combine results from both CV approaches
cv_standard_df = pd.DataFrame(linreg_cv_standard).assign(
    Approach="Grouping Ignored"
)
cv_grouped_df = pd.DataFrame(linreg_cv_grouped).assign(
    Approach="Grouping Respected"
)

cv_results = (
    pd.concat([cv_standard_df, cv_grouped_df])
    .groupby(["Approach"])[["test_rsq", "test_rmse"]]
    .agg([np.mean, std_err])
)

print("\nComparison of CV approaches:")
print(cv_results)
# - Ignoring grouping leads to OVERESTIMATION of model performance
# - Ignoring grouping leads to UNDERESTIMATION of uncertainty (smaller SE)
# - The grouped approach gives more realistic estimates!

# True OOS Performance --------------------------------------------------

# Train models using both splitting approaches
linreg_standard = clone(linreg_pipe).fit(
    ht_train_standard[predictors], ht_train_standard[outcome]
)
linreg_grouped = clone(linreg_pipe).fit(ht_train[predictors], ht_train[outcome])

# Get predictions on test set
y_pred_standard = linreg_standard.predict(ht_test[predictors])
y_pred_grouped = linreg_grouped.predict(ht_test[predictors])

# Calculate performance metrics
print("\nTest set performance (Grouping Ignored):")
print(f"  Rsq: {r2_score(ht_test[outcome], y_pred_standard):.4f}")
print(
    f"  RMSE: {root_mean_squared_error(ht_test[outcome], y_pred_standard):.4f}"
)

print("\nTest set performance (Grouping Respected):")
print(f"  Rsq: {r2_score(ht_test[outcome], y_pred_grouped):.4f}")
print(
    f"  RMSE: {root_mean_squared_error(ht_test[outcome], y_pred_grouped):.4f}"
)
# Again: ignoring grouping overestimates true OOS performance!

# Create dataframe for plotting predictions
n_pred = len(y_pred_standard)
pred_df = pd.DataFrame(
    {
        "predicted": np.concatenate([y_pred_standard, y_pred_grouped]),
        "actual": np.concatenate(
            [ht_test[outcome].values, ht_test[outcome].values]
        ),
        "approach": (
            ["Grouping Ignored"] * n_pred + ["Grouping Respected"] * n_pred
        ),
    }
)

# Get min and max for identity line
min_val = ht_test[outcome].min()
max_val = ht_test[outcome].max()

p_predictions = (
    ggplot(pred_df, aes(x="predicted", y="actual", color="approach"))
    + geom_point(alpha=0.6, size=2)
    + geom_abline(intercept=0, slope=1, color="grey", linetype="dashed", size=1)
    + labs(
        x="Predicted Average Daily Rate",
        y="Actual Average Daily Rate",
        title="Predicted vs Actual: Comparing Split Approaches",
        color="Split Approach",
    )
    + scale_color_manual(
        values={
            "Grouping Ignored": "skyblue",
            "Grouping Respected": "lightcoral",
        }
    )
    + theme_minimal()
    + theme(figure_size=(10, 8))
)

p_predictions.draw(show=True)
