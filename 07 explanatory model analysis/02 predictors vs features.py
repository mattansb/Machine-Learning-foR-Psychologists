from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn import set_config

import dalex as dx

from ISLP import load_data

# Configure sklearn to output pandas DataFrames
set_config(transform_output="pandas")

# Fit a model ---------------------------------------------
# Let's fit a regression model!

Hitters = load_data("Hitters")
Hitters = Hitters.dropna(subset=["Salary"])

# split the data
outcome = "Salary"
features = Hitters.columns.difference([outcome])

X_train, X_test, y_train, y_test = train_test_split(
    Hitters[features], Hitters[outcome], train_size=0.7, random_state=20251201
)


# Preprocessing steps:


# Custom transformer for Interaction Hits:Years
def interaction_hits_years(df):
    df_out = df.copy()
    df_out["Hits_x_Years"] = df_out["center__Hits"] * df_out["center__Years"]
    return df_out


# Custom transformer for Poly HmRun degree 2:
def poly_hmrun(df):
    df_out = df.copy()
    df_out["HmRun_poly_2"] = df_out["center__HmRun"] ** 2
    return df_out


# Column selection
s_categorical_features = make_column_selector(
    dtype_include=["category", object]
)

ct = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(sparse_output=False),
            s_categorical_features,
        ),
        ("center", StandardScaler(with_std=False), ["Hits", "Years", "HmRun"]),
    ],
    remainder="passthrough",
)

preprocessor = Pipeline(
    steps=[
        ("ct", ct),
        ("interact", FunctionTransformer(interaction_hits_years)),
        ("poly", FunctionTransformer(poly_hmrun)),
    ]
)

# fit an RF with mtry = 7
rf_spec = RandomForestRegressor(
    n_estimators=500, max_features=7, min_samples_leaf=5, random_state=20260125
)

rf_pipe = Pipeline([("preprocessor", preprocessor), ("model", rf_spec)])
rf_pipe.fit(X_train, y_train)


# Explain predictors  -----------------------------------------------
# This is the default behavior:
# 1. Input to explainer: raw data
# 2. Model builds features and make predictons

rf_xpln_predictors = dx.Explainer(
    rf_pipe, label="predictors", data=X_train, y=y_train
)

## Explain a single prediction ------------------------------
# Let's explain obs 61:
obs_61 = X_test.iloc[[61], :]
shap_61_predictors = rf_xpln_predictors.predict_parts(
    X_test.iloc[[61], :], type="shap"
)


### Variable importance ----------------------

vi_perm_predictors = rf_xpln_predictors.model_parts(
    B=10,  # Number of permutations
    variable_groups=None,
)


# Explain features  -----------------------------------------------

# But we can also explain the features:
preprocessor_trained = rf_pipe.named_steps["preprocessor"]
model_trained = rf_pipe.named_steps["model"]

X_train_processed = preprocessor_trained.transform(X_train)

rf_xpln_features = dx.Explainer(
    model_trained, label="features", data=X_train_processed, y=y_train
)


## Explain a single prediction ------------------------------
# Let's explain obs 61 again - note that these are (and must!) be the same:

obs_61_processed = preprocessor_trained.transform(obs_61)

print(rf_pipe.predict(obs_61))  # prediction from predictors model
print(model_trained.predict(obs_61_processed))  # from features model

shap_61_features = rf_xpln_features.predict_parts(obs_61_processed, type="shap")

shap_61_features.plot(objects=shap_61_predictors, max_vars=100)
# Note HmRun_poly_2 and Hits_x_Years are features, not predictors!


### Variable importance ----------------------

# Note that permutation of features might not make sense in some cases - for
# example, HmRun and HmRun^2 are dependent features, as are Hits_x_Years and
# Hits / Years so permuting one but not the others might not make sense.
vi_perm_features = rf_xpln_features.model_parts(
    B=10,  # Number of permutations
    variable_groups=None,
)

vi_perm_features.plot(objects=vi_perm_predictors, bar_width=5, max_vars=100)
