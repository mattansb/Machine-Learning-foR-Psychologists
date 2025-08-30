import pandas as pd
import numpy as np
from plotnine import *

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor

from ISLP import load_data

# The data -----------------------------------------------------------

Auto = load_data("Auto")
Auto["cylinders"] = pd.Categorical(Auto["cylinders"])
Auto["origin"] = pd.Categorical(Auto["origin"])

## Data splitting ----------------------------------------
# We will be using data a little differently here since we will be doing 3
# things along the way:
# 1. Tune hyper parameters
# 2. Compare models (and select between them)
# 3. Test the final model
# We want a "clean" dataset for each type of work that is independent from the
# data used in previous steps.

# So let's split the data in 3 parts:
# 35% for tuning, 35% for comparing, and the remaining (30%) for testing:
p = (0.35, 0.35)

Auto_train_compare, Auto_test = train_test_split(
    Auto, test_size=(1 - np.sum(p)), random_state=1
)

Auto_train, Auto_compare = train_test_split(
    Auto_train_compare, test_size=(p[0] / np.sum(p)), random_state=1
)

# As you can see, these splits require a lot of data.
# In smaller samples we might use loo-cv for comparing models, or collapse
# decision nodes.


## Get resampled results --------------------------------------------------

### Model 1: Linear regression --------------------------------------------

outcome = "mpg"
features1 = ["horsepower", "weight"]

linreg = Pipeline(steps=[("regressor", LinearRegression())])

# (No tuning needed.)

# Split comparison data (why?):
cv_compare = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold CV
# We will use these folds for ALL the models - then we can compare the models'
# performance on a fold-wise basis!

# These are the metrics we're interested in:
reg_scorer = {
    "r2": make_scorer(r2_score),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
}

# We will use the cross_validate() function. This function doesn't actually
# return a fitted model - it just computes a set of performance metrics across
# the resamples.
linreg1_oos = cross_validate(
    estimator=linreg,
    X=Auto_compare[features1],  # Use Auto_compare data for OOS evaluation
    y=Auto_compare[outcome],
    cv=cv_compare,  # Use the SAME folds for comparison
    scoring=reg_scorer,
    n_jobs=-1,
)

# get a summary of the metrics across resamples:
linreg1_oos_df = pd.DataFrame(linreg1_oos)

print(
    linreg1_oos_df[["test_r2", "test_mae"]]
    .agg(["mean", lambda x: np.std(x) / np.sqrt(len(x))])
    .rename(index={"<lambda>": "SE"})
    .T
)
# Why is MAE negative???
# Because setting greater_is_better=False makes the score reversed. (Silly)


### Model 2: Linear regression --------------------------------------------

# Use all features!
features2 = Auto_train.columns.difference([outcome])

linreg2_oos = cross_validate(
    estimator=linreg,
    X=Auto_compare[features2],  # Use Auto_compare data for OOS evaluation
    y=Auto_compare[outcome],
    cv=cv_compare,  # Use the SAME folds for comparison
    scoring=reg_scorer,
    n_jobs=-1,
)

### Model 3: KNN --------------------------------------------

prepoc_cat = ColumnTransformer(
    transformers=[
        ("dummy", OneHotEncoder(sparse_output=False), ["origin", "cylinders"])
    ]
)

preprocessor = Pipeline(
    steps=[("dummy_cats", prepoc_cat), ("z", StandardScaler())]
)

knn_pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("knn", KNeighborsRegressor())]
)

#### Tune the model --------------------------

cv_tune = KFold(n_splits=5)
# Using k=5, but more importantly we're using different data for tuning!

knn_grid = {"knn__n_neighbors": np.linspace(10, 100, 7, dtype=int)}

knn_tuner = GridSearchCV(
    estimator=knn_pipe,
    param_grid=knn_grid,
    cv=cv_tune,  # Use separate tuning folds
    scoring=reg_scorer,
    refit="mae",  # Select best based on MAE (negative MAE for GridSearchCV)
    verbose=0,
    n_jobs=-1,
)

knn_tuner.fit(Auto_train[features2], Auto_train[outcome])


cv_res = pd.DataFrame(knn_tuner.cv_results_)

# Select by one-SE rule:
from one_se_rule import *

low_complexity_k = make_one_se_rule_selector(
    n_fold=5,
    by=["param_knn__n_neighbors"],
    ascending=False,
    scorer="mae",
    greater_is_better=True,
    ret_index=False,
)
params_oneSE = low_complexity_k(cv_res)

print(params_oneSE)
cv_res.plot(x="param_knn__n_neighbors", y="mean_test_mae", kind="line")

knn_pipe.set_params(
    knn__n_neighbors=int(params_oneSE.param_knn__n_neighbors.iloc[0])
)

# And finally, resampling results with the Auto_compare data:
knn_oos = cross_validate(
    estimator=knn_pipe,
    X=Auto_compare[features2],  # Use Auto_compare data for OOS evaluation
    y=Auto_compare[outcome],
    cv=cv_compare,  # Use the SAME folds for comparison
    scoring=reg_scorer,
    n_jobs=-1,
)

## Compare the resampled performance ------------------------------------------------
# We're actually doing two things with this:
# 1. We are getting CV estimates of the OOS performance of the models.
# 2. Because we used the *same* folds, we can compare the models in a paired
#    fashion!

cv_results = pd.concat(
    [
        linreg1_oos_df.assign(model="linear1"),
        pd.DataFrame(linreg2_oos).assign(model="linear2"),
        pd.DataFrame(knn_oos).assign(model="KNN"),
    ]
)

cv_results["model"] = pd.Categorical(
    cv_results["model"], categories=["linear1", "linear2", "KNN"]
)

cv_results["fold"] = cv_results.index

cv_results = cv_results.melt(
    id_vars=["model", "fold"],
    value_vars=["test_r2", "test_mae"],
    var_name="metric",
    value_name="score",
)

cv_summary = (
    cv_results.groupby(["model", "metric"])
    .agg(
        mean=("score", "mean"),
        se=("score", lambda x: np.std(x) / np.sqrt(len(x))),
    )
    .reset_index()
)


# Let's make a pretty plot!
(
    ggplot(cv_results, aes("model", "score"))
    + facet_wrap("metric", scales="free_y")
    # fold-data
    + geom_line(aes(group="fold"))
    # summary
    + geom_pointrange(
        aes(y="mean", ymin="mean - se", ymax="mean + se"),
        data=cv_summary,
        color="red",
    )
).show()

### Contrast ----------------

cv_compareare_lin2KNN = (
    cv_results.pivot(index=["fold", "metric"], columns="model", values="score")
    .reset_index()
    .assign(diff=lambda x: x["linear2"] - x["KNN"])
    .groupby("metric")
    .agg(
        mean_diff=("diff", "mean"),
        se_diff=("diff", lambda x: np.std(x) / np.sqrt(len(x))),
    )
    .reset_index()
)


for i in cv_compareare_lin2KNN.index:
    met, M, SE = tuple(
        cv_compareare_lin2KNN.loc[i, ["metric", "mean_diff", "se_diff"]]
    )
    lb = M - 1.96 * SE
    ub = M + 1.96 * SE
    print(f"{met}: A diff of {M:.3f}, 95% CI [{lb:.3f}, {ub:.3f}]")

# Investigating prediction errors ---------------------------------------------
# It can often be interesting to see not only which model is better, but also
# where different models fail.

# Let's train the selected model(s?) on tune+compare datasets
linreg.fit(Auto_train_compare[features1], Auto_train_compare[outcome])
knn_pipe.fit(Auto_train_compare[features2], Auto_train_compare[outcome])


# Note we're using the test data
y_pred1 = linreg.predict(Auto_test[features1])
y_pred2 = knn_pipe.predict(Auto_test[features2])

## Sub-sample performance -------------------------------------

# We can also look are group performance indices:
Auto_test.assign(
    pred=y_pred1, big_mpg=Auto_test["mpg"] >= Auto_test["mpg"].median()
).groupby(["origin", "big_mpg"]).apply(
    lambda x: pd.Series(
        mean_absolute_error(x[outcome], x["pred"]), index=["MAE"]
    )
)

(
    ggplot(
        Auto_test.assign(pred=y_pred1),
        aes("origin", "mpg - pred", fill="mpg >= mpg.median()"),
    )
    + geom_hline(yintercept=0)
    + geom_violin()
).show()
# We can seen that the model tends to fail the most for cars from Europe with
# high MPG, and least for American cars with low MPG.


# etc...
Auto_test.assign(
    pred=y_pred1, hp_cat=pd.qcut(Auto_test["horsepower"], 3)
).groupby(["hp_cat"]).apply(
    lambda x: pd.Series(
        mean_absolute_error(x[outcome], x["pred"]), index=["MAE"]
    )
)

## Comparing error distributions --------------------------------
(
    ggplot(mapping=aes("mpg - pred", fill="model"))
    + geom_vline(xintercept=0)
    + geom_density(
        color=None,
        alpha=0.6,
        data=Auto_test.assign(pred=y_pred1, model="Linear Reg\n(simple)"),
    )
    + geom_density(
        color=None,
        alpha=0.6,
        data=Auto_test.assign(pred=y_pred2, model="KNN"),
    )
    + labs(x=r"$mpg - \hat{mpg}$ (Prediction Error)")
).show()
# It seems that the KNN model gives more negative errors (tends to overestimate
# mpg).

# For classification problems, we can also compare errors with
import statsmodels.stats as sms

help(sms.contingency_tables.mcnemar)
