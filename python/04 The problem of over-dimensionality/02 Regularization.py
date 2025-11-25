from ISLP import load_data

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


# Hitters DATA and the PROBLEM ---------------------------------

# Hitters Dataset: Baseball Data from the 1986 and 1987 seasons
Hitters = load_data("Hitters")
Hitters.dropna(subset="Salary", inplace=True)
# A data frame with 263 observations of major league players on the following 20
# variables.

print(Hitters.shape)
print(Hitters.columns)

# We wish to predict a baseball player's *Salary* on the basis of preformence
# variables in the previous year.
# Which 19 predictors will be best for predicting Salary?

outcome = "Salary"
features = Hitters.columns.difference([outcome])  # use all columns as features

# Split:
X_train, X_test, y_train, y_test = train_test_split(
    Hitters[features], Hitters[outcome], train_size=0.7, random_state=123442
)
# Our data is REALLY SMALL such that splitting the data to train and test might
# leave us with very small datasets.


ct = ColumnTransformer(
    transformers=[
        # - origin is a factor, so we'll produce dummy coding
        (
            "dummy",
            OneHotEncoder(drop=None, sparse_output=False),
            ["Division", "League", "NewLeague"],
        ),
    ],
    remainder="passthrough",
).set_output(transform="pandas")


def f_interact(x):
    """
    This function takes a data frame, then adds a column
    that is the product of and remainder__horsepower
    """
    x["int1"] = x["remainder__HmRun"] * x["dummy__League_A"]
    x["int2"] = x["remainder__HmRun"] * x["dummy__League_N"]
    return x


preprocessor = Pipeline(
    steps=[
        ("coltrans", ct),
        ("int", FunctionTransformer(f_interact)),
        ("scale", StandardScaler()),
    ]
)

# Ridge Regression -------------------------

# We will perform ridge regression and the lasso in order to predict Salary on
# the Hitters data.

ridge_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("ridge", Ridge())]
)
# We have one hyperparameter - the PENALTY (lambda, here called alpha for some
# reason).


## Tune ------------------------------------

# Let's tune the PENALTY.
ridge_grid = {"ridge__alpha": np.logspace(-2, 7, 20)}

# Using 5-fold CV:
cv_5folds = KFold(n_splits=5)

# Tune the model
grid_search =   (
    estimator=ridge_pipeline,
    param_grid=ridge_grid,
    cv=cv_5folds,
    scoring=["rmse"],
    refit=True,
    verbose=1,  # Show progress
    n_jobs=-1,  # Use all available CPU cores
)

grid_search.fit(X_train, y_train)

cv_res = pd.DataFrame(grid_search.cv_results_)

cv_res.plot(x="param_ridge__alpha", y="mean_test_auc", kind="line")
cv_res.plot(x="param_ridge__alpha", y="mean_test_accuracy", kind="line")

print(grid_search.best_estimator_)

GridSearchCV
