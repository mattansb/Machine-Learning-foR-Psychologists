import numpy as np
import pandas as pd
from plotnine import *

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    cross_val_predict,
    KFold,
    GridSearchCV,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn import set_config

from ISLP import load_data

set_config(display="diagram")


# Read more about model comparisons using resampling here:
# https://www.tmwr.org/compare
# The basic idea behind model comparison with k-folds, is that we can measure
# the OOS performance of each model on each of the folds. Some folds might (by
# chance) be harder / easier to predict for across models, so these are paired
# measures of fit, but each fold is independent (insofar as they _are_ part of a
# set) so we can use rather standard procedures for comparing models across
# paired samples.

# The data -----------------------------------------------------------

Auto = load_data("Auto")
help(Auto)
Auto["cylinders"] = pd.Categorical(Auto["cylinders"])
Auto["origin"] = pd.Categorical(Auto["origin"])

## Data splitting ----------------------------------------

# We will be using the training set for both tuning (within-model comparison) and
# model comparison (between-model comparison). The test set will be used only at
# the end to get a final estimate of selected model performance.
X = Auto.drop(columns=["mpg"])
y = Auto["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=20251202
)

## Get resampled results --------------------------------------------------

### Model 1: Linear regression --------------------------------------------

# Simple linear regression with two predictors
features_1 = ["horsepower", "weight"]
X_train_1 = X_train[features_1]
X_test_1 = X_test[features_1]

linreg_spec = LinearRegression()

linreg1_pipeline = Pipeline(steps=[("regressor", linreg_spec)])

# (No tuning needed.)

# Split data:
cv_compare = KFold(n_splits=10, shuffle=True, random_state=20251201)
# We will use these folds for ALL the models - then we can compare the models'
# performance on a fold-wise basis!

# These are the metrics we're interested in:
scoring = {
    "r2": make_scorer(r2_score),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
}

# We will use cross_validate() function. This function doesn't actually
# return a fitted model - it just computes a set of performance metrics across
# the resamples.
linreg1_oos = cross_validate(
    linreg1_pipeline, X_train_1, y_train, cv=cv_compare, scoring=scoring
)

print("\nModel 1 (Linear Regression - simple):")
linreg1_results = pd.DataFrame(linreg1_oos)
linreg1_results["fold"] = range(0, 10)
print(linreg1_results)

### Model 2: Linear regression --------------------------------------------

# More complex linear regression with all predictors
cat_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

preprocessor_2 = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first", sparse_output=False, handle_unknown="ignore"
            ),
            cat_cols,
        ),
    ],
    remainder="passthrough",
)

# (No tuning needed.)

linreg2_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor_2), ("regressor", linreg_spec)]
)

linreg2_oos = cross_validate(
    linreg2_pipeline,
    X_train,
    y_train,
    cv=cv_compare,  # We are using the SAME resamples!
    scoring=scoring,
)

print("\nModel 2 (Linear Regression - Complex):")
linreg2_results = pd.DataFrame(linreg2_oos)
linreg2_results["fold"] = range(0, 10)
print(linreg2_results)


### Model 3: KNN --------------------------------------------

ct_3 = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first", sparse_output=False, handle_unknown="ignore"
            ),
            cat_cols,
        )
    ]
)

preprocessor_3 = Pipeline(
    steps=[
        ("dummy", ct_3),
        ("scaler", StandardScaler()),
    ]
)

knn_spec = KNeighborsRegressor()

knn_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor_3), ("regressor", knn_spec)]
)


#### Tune the model --------------------------

param_grid = {"regressor__n_neighbors": [5, 15, 25, 35, 50, 75, 100]}

# Note we're using a different set of splits! This is a single k-fold CV which
# can protect against overfitting during model selection later.
cv_tune = KFold(n_splits=10, shuffle=True, random_state=20251202)

knn_tuner = GridSearchCV(
    knn_pipeline,
    param_grid=param_grid,
    cv=cv_tune,
    scoring=scoring,
    refit="mae",
    n_jobs=-1,
    return_train_score=False,
)

knn_tuner.fit(X_train, y_train)

# Visualize tuning results
results_knn = pd.DataFrame(knn_tuner.cv_results_)
results_knn_long = results_knn.melt(
    id_vars=["param_regressor__n_neighbors"],
    value_vars=["mean_test_r2", "mean_test_mae"],
    var_name="metric",
    value_name="score",
)

# Invert MAE for plotting (so higher is better)
results_knn_long["score"] = results_knn_long.apply(
    lambda row: -row["score"]
    if row["metric"] == "mean_test_mae"
    else row["score"],
    axis=1,
)

# Create readable metric labels
results_knn_long["metric_label"] = results_knn_long["metric"].map(
    {"mean_test_r2": "R-square", "mean_test_mae": "MAE"}
)

p_tune = (
    ggplot(results_knn_long, aes(x="param_regressor__n_neighbors", y="score"))
    + geom_line(size=1)
    + geom_point(size=3)
    + facet_wrap("~ metric_label", scales="free_y")
    + labs(
        x="K (neighbors)", y="Score", title="Tuning Results: Performance vs K"
    )
    + theme_minimal()
    + theme(figure_size=(12, 4))
)

p_tune.draw(show=True)

# Selecting by the one-SE rule also protects us from overfitting:
from one_se_rule import *

low_complexity_k = make_one_se_rule_selector(
    n_fold=10,
    by=["param_regressor__n_neighbors"],
    ascending=False,
    scorer="mae",
    ret_index=False,
)

params_oneSE = low_complexity_k(results_knn)
print(f"\none-SE parameters: {list(params_oneSE.param_regressor__n_neighbors)}")

#### Get OOS results -----------------------

knn_pipeline.set_params(
    regressor__n_neighbors=int(
        params_oneSE.param_regressor__n_neighbors.iloc[0]
    )
)

knn_oos = cross_validate(
    knn_pipeline,
    X_train,
    y_train,
    # Note we're using the same splits as models 1 and 2!
    cv=cv_compare,
    scoring=scoring,
)

print("\nModel 3 (KNN):")
knn_results = pd.DataFrame(knn_oos)
knn_results["fold"] = range(0, 10)
print(linreg2_results)


## Compare the resampled performance ------------------------------------------------
# We're actually doing two things with this:
# 1. We are getting CV estimates of the OOS performance of the models.
# 2. Because we used the *same* folds, we can compare the models in a paired
#    fashion!

### Plot -----------------

# Combine all results
cv_results = pd.concat(
    [
        linreg1_results.assign(model="linear1"),
        linreg2_results.assign(model="linear2"),
        knn_results.assign(model="KNN"),
    ]
)

# Reshape for plotting
cv_results_long = pd.melt(
    cv_results,
    id_vars=["fold", "model"],
    value_vars=["test_r2", "test_mae"],
    var_name="metric",
)


# Summary statistics
def std_err(x):
    return x.std() / np.sqrt(len(x))


cv_summary = (
    cv_results_long.groupby(["model", "metric"])["value"]
    .agg(["mean", std_err])
    .reset_index()
)

# Plot:
p = (
    ggplot(cv_results_long, aes(x="model", y="value"))
    + facet_wrap("~ metric", scales="free_y")
    # fold-data
    + geom_line(
        aes(group="factor(fold)"),
        size=0.5,
    )
    # Summary points with error bars
    + geom_pointrange(
        aes(
            y="mean",
            ymin="mean - std_err",
            ymax="mean + std_err",
            color="model",
        ),
        data=cv_summary,
    )
    + labs(x="Model", y="Value", title="Cross-validation results across folds")
    + theme_minimal()
    + theme(figure_size=(14, 5))
)

p.draw(show=True)


### Contrast ----------------

# Calculate paired differences
cv_compare_lin2_knn = pd.DataFrame(
    {
        "fold": range(0, 10),
        "r2_linear2": linreg2_results["test_r2"],
        "r2_KNN": knn_results["test_r2"].values,
        "mae_linear2": linreg2_results["test_mae"],
        "mae_KNN": knn_results["test_mae"].values,
    }
)

cv_compare_lin2_knn["diff_r2"] = (
    cv_compare_lin2_knn["r2_linear2"] - cv_compare_lin2_knn["r2_KNN"]
)
cv_compare_lin2_knn["diff_mae"] = (
    cv_compare_lin2_knn["mae_linear2"] - cv_compare_lin2_knn["mae_KNN"]
)

# Summary of differences
print("\n" + "=" * 60)
print("Comparison: Linear2 vs KNN")
print("=" * 60)
for metric in ["r2", "mae"]:
    diff_col = f"diff_{metric}"
    mean_diff = cv_compare_lin2_knn[diff_col].mean()
    se_diff = cv_compare_lin2_knn[diff_col].std() / np.sqrt(10)
    lb = mean_diff - 1.96 * se_diff
    ub = mean_diff + 1.96 * se_diff

    print(
        f"A diff of {mean_diff:.4f} in {metric.upper()}, 95% CI[{lb:.4f}, {ub:.4f}]"
    )

# Other uses for the resampled results --------------------------------

# It can often be interesting to see not only which model is better, but also
# where different models fail.

# scikit-learn's cross_validate doesn't return individual predictions by
# default. We can use cross_val_predict to get OOS predictions:

linreg2_predictions = cross_val_predict(
    linreg2_pipeline, X_train, y_train, cv=cv_compare
)

knn_predictions = cross_val_predict(
    knn_pipeline, X_train, y_train, cv=cv_compare
)

## 1. Comparing error distributions --------------------------------

errors_linreg2 = y_train.values - linreg2_predictions
errors_knn = y_train.values - knn_predictions

# Create dataframe for plotting
errors_df = pd.DataFrame(
    {
        "error": np.concatenate([errors_linreg2, errors_knn]),
        "model": ["Linear Reg\n(complex)"] * len(errors_linreg2)
        + ["KNN"] * len(errors_knn),
    }
)

p_errors = (
    ggplot(errors_df, aes(x="error", fill="model"))
    + geom_histogram(
        aes(y=after_stat("density")), bins=30, alpha=0.6, position="identity"
    )
    + geom_vline(xintercept=0, color="black", linetype="solid", size=1)
    + labs(
        x="mpg - $\\hat{mpg}$",
        y="Density",
        title="Comparing Error Distributions",
    )
    + scale_fill_manual(
        values={"Linear Reg\n(complex)": "#ff7f0e", "KNN": "#2ca02c"}
    )
    + theme_minimal()
    + theme(figure_size=(10, 6))
)

p_errors.draw(show=True)
# It seems that the KNN gives more negative errors (tends to overestimate mpg).

# Have we selected a model?
# Time to see how it performs on the test set!
# ...

## 2. Sub-sample performance -------------------------------------
# Slicing / fairness analyses

# We can also look at group performance indices:

# Add the original variables back in:
train_with_preds = X_train.copy()
train_with_preds["mpg"] = y_train.values
train_with_preds["pred_linreg2"] = linreg2_predictions
train_with_preds["pred_knn"] = knn_predictions

# Group by origin and high/low mpg
train_with_preds["high_mpg"] = (
    train_with_preds["mpg"] >= train_with_preds["mpg"].median()
)

group_performance = (
    train_with_preds.groupby(["origin", "high_mpg"])
    .apply(
        lambda x: r2_score(x["mpg"], x["pred_linreg2"]), include_groups=False
    )
    .reset_index(name="r2_linreg2")
)
print("\nR² by origin and MPG level (Linear Reg 2):")
print(group_performance.sort_values("r2_linreg2"))

# Visualize errors by group
train_with_preds["error_linreg2"] = (
    train_with_preds["mpg"] - train_with_preds["pred_linreg2"]
)

p_errors_by_group = (
    ggplot(
        train_with_preds, aes(x="origin", y="error_linreg2", fill="high_mpg")
    )
    + geom_violin(alpha=0.7)
    + geom_hline(yintercept=0, color="black", linetype="dashed", size=1)
    + labs(
        x="Origin",
        y="mpg - $\\hat{mpg}$",
        title="Prediction Errors by Origin and MPG Level",
        fill="High MPG",
    )
    + theme_minimal()
    + theme(figure_size=(10, 6))
)

p_errors_by_group.draw(show=True)
# We can see that the model tends to fail the most for cars from Europe with
# high MPG, and least for American cars with low MPG.

# etc...
# Group by horsepower tertiles
train_with_preds["hp_group"] = pd.qcut(
    train_with_preds["horsepower"], 3, labels=["Low", "Med", "High"]
)
group_performance_hp = (
    train_with_preds.groupby("hp_group")
    .apply(
        lambda x: r2_score(x["mpg"], x["pred_linreg2"]), include_groups=False
    )
    .reset_index(name="r2_linreg2")
)
print("\nR² by horsepower group (Linear Reg 2):")
print(group_performance_hp)
