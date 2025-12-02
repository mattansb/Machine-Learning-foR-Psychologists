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
    #     mean_absolute_error,
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

# Typically, we would use
ht_trainX, ht_testX = train_test_split(
    hotel_rates, train_size=(2 / 3), random_state=111
)

# When this would mean we have some information from the test countries in our
# training set:
print(any(ht_testX["country"].isin(ht_trainX["country"])))

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py


# Instead, we will use a grouped split:
splitter_grouped = GroupShuffleSplit(
    n_splits=1, train_size=(2 / 3), random_state=111
)

train_idx_g, test_idx_g = next(
    splitter_grouped.split(hotel_rates, groups=hotel_rates["country"])
)

ht_train = hotel_rates.iloc[train_idx_g].copy()
ht_test = hotel_rates.iloc[test_idx_g].copy()

# Non of the country in the test set appear in the training set!
print(any(ht_test["country"].isin(ht_train["country"])))


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

## Standard CV -----------------------------

folds_standard = KFold(n_splits=5, shuffle=True, random_state=111)

linreg_cv_standard = cross_validate(
    estimator=linreg_pipe,
    X=ht_train[predictors],
    y=ht_train[outcome],
    cv=folds_standard,
    scoring=scorers_reg,
    return_train_score=False,
    n_jobs=-1,
)
# Data from each country are now scattered across folds.

## Grouped CV -----------------------------

folds_g = GroupKFold(n_splits=5, shuffle=True, random_state=111)

linreg_cv_g = cross_validate(
    estimator=linreg_pipe,
    X=ht_train[predictors],
    y=ht_train[outcome],
    cv=folds_g,
    groups=ht_train["country"],
    scoring=scorers_reg,
    return_train_score=False,
    n_jobs=-1,
)


## Compare -------------------------------


def std_err(x):
    return x.std() / np.sqrt(len(x))


cv_results = (
    pd.concat(
        [
            pd.DataFrame(linreg_cv_standard).assign(Group="Ignored"),
            pd.DataFrame(linreg_cv_g).assign(Group="Accounted for"),
        ]
    )
    .groupby(["Group"])[["test_rsq", "test_rmse"]]
    .agg([np.mean, std_err])
)
print(cv_results)
# When ignoring the grouping the data we both over estimate the models
# performance with CV, but we're also over confident in the out-of-sample
# performance by ignoring the dependency (looks at the difference in the std_err
# of our metrics)!

# True OOS Performance --------------------------------------------------

linreg_mod = clone(linreg_pipe).fit(ht_trainX[predictors], ht_trainX[outcome])
linreg_mod_g = clone(linreg_pipe).fit(ht_train[predictors], ht_train[outcome])

y_pred = linreg_mod.predict(ht_test)
y_pred_g = linreg_mod_g.predict(ht_test)

r2_score(ht_test[outcome], y_pred)
r2_score(ht_test[outcome], y_pred_g)

root_mean_squared_error(ht_test[outcome], y_pred)
root_mean_squared_error(ht_test[outcome], y_pred_g)
# Again - we are over estimating our out of sample performance :(

# Create dataframe for plotting
pred_df = pd.DataFrame(
    {
        "predicted": np.concatenate([y_pred, y_pred_g]),
        "actual": np.concatenate(
            [ht_test[outcome].values, ht_test[outcome].values]
        ),
        "grouping": ["Ignored Grouping"] * len(y_pred)
        + ["Accounted for Grouping"] * len(y_pred_g),
    }
)

# Get min and max for identity line
min_val = ht_test[outcome].min()
max_val = ht_test[outcome].max()

p_predictions = (
    ggplot(pred_df, aes(x="predicted", y="actual", color="grouping"))
    + geom_point(alpha=0.6, size=2)
    + geom_abline(intercept=0, slope=1, color="grey", linetype="dashed", size=1)
    + labs(
        x="Predicted",
        y="Actual",
        title="Predicted vs Actual Values",
        color="Grouping",
    )
    + scale_color_manual(
        values={
            "Ignored Grouping": "skyblue",
            "Accounted for Grouping": "lightcoral",
        }
    )
    + theme_minimal()
    + theme(figure_size=(10, 8))
)

p_predictions.draw(show=True)
