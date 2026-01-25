import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import *

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn import set_config

import dalex as dx

from ISLP import load_data
from palmerpenguins import load_penguins

# Configure sklearn to output pandas DataFrames
set_config(transform_output="pandas")

# Regression ------------------------------------------------------------------

## Fit a model ---------------------------------------------
# Let's fit a regression model!

Hitters = load_data("Hitters")
# Dropping NAs in Salary
Hitters = Hitters.dropna(subset=["Salary"])
print(Hitters.info())

# split the data
outcome = "Salary"
features = Hitters.columns.difference([outcome])

X_train, X_test, y_train, y_test = train_test_split(
    Hitters[features], Hitters[outcome], train_size=0.7, random_state=1111
)


# We need to defining a custom transformer to make interaction columns...
def f_interact(x):
    """
    This function takes a data frame, then adds a column
    that is the product of Walks and PutOuts
    """
    x_out = x.copy()
    x_out["int"] = x_out["remainder__PutOuts"] * x_out["remainder__Walks"]
    return x_out


ct_dummy = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(sparse_output=False),
            ["Division", "NewLeague", "League"],
        )
    ],
    remainder="passthrough",
)

# Defining the preprocessing steps
preprocessor = Pipeline(
    [
        # 1. Encode categoricals (and pass through numerics)
        ("encoder", ct_dummy),
        # 2. Add interaction
        ("interact", FunctionTransformer(f_interact)),
        # 3. Normalize all numerics
        ("scaler", StandardScaler()),
    ]
)

# We will be using 5 nearest neighbors:
knn = KNeighborsRegressor(n_neighbors=5)

knn_pipe = Pipeline([("preprocessor", preprocessor), ("model", knn)])
knn_pipe.fit(X_train, y_train)


## Explain the model -----------------------------------------------
# All the methods we'll be using are model agnostic - that means that they work
# for ANY model that can take ANY data and generate a prediction.
# See:
# https://www.tmwr.org/explain
# https://ema.drwhy.ai/

# We first need to setup an explainer:
knn_xplnr = dx.Explainer(knn_pipe, label="KNN (K=5)", data=X_train, y=y_train)


### Explain a single prediction ------------------------------
# Models make predictions. For example, we can see that our model predicts for
# the obs. 44 a salary of 1112 (*1000 = 1,112,000$)
print(knn_pipe.predict(X_test.iloc[[44], :]))


# But why?
# What is it about this row that leads our model to make such a prediction?
y_train.mean()


# We will be using SHAP (SHapley Additive exPlanations) to try and answer this
# question SHAP is a method that tries to estimate what each predictor
# contributes to each single prediction - accounting for whatever interactions
# or conditional effects it might have. The ideas are based on game theory -
# really interesting stuff. Lots of math.
#
# In essence, SHAP tries to attribute a predictions deviation from the mean
# prediction to the various predictors.

shap_44 = knn_xplnr.predict_parts(X_test.iloc[[44], :], type="shap")

shap_44.plot()
# - The contributions sum to the predicted values over the baseline (mean).
# - The box plots represent the distributions of contributions.
#   See: https://ema.drwhy.ai/shapley.html

shap_44.plot(max_vars=100)  # get more features


# Note that predictions can be attributed differently between predictors
# across different predictions. Let's compare obs. 44 to obs. 34:
shap_34 = knn_xplnr.predict_parts(X_test.iloc[[34], :], type="shap")

shap_44.plot(objects=shap_34, max_vars=100)


# Note that SHAP analysis DOES NOT give us any counterfactual information - we
# don't know that if 44 was in a different Division his salary would have been
# higher - just that the model chose to give him a lower prediction because he's
# in division E.
# In other words - we are explaining the model's predictions, but we are not
# explaining the salary! There is no causal information here, we are NOT in
# "explaining mode", we are still in "prediction mode"!

# Alternatives:
# - Local Interpretable Model-agnostic Explanations (LIME):
#   https://lime.data-imaginist.com
#   https://ema.drwhy.ai/LIME.html
#   These are useful for models with many (hundreds-thousands+) predictors
# - And more https://ema.drwhy.ai/InstanceLevelExploration.html


### Variable importance ----------------------
# We've already seen model-based variable importance methods - but there are
# also model-agnostic methods, such as Permutation-based variable importance.
# See: https://ema.drwhy.ai/featureImportance.html

# This method assesses a predictors's contribution to a model's predictive
# accuracy by randomly shuffling a predictor's values - breaking its
# relationship with the outcome. The model's performance is then evaluated on
# this new permuted data.
# If the variable has no contribution, permuting it will have little to no
# effect on the model's performance, while variables with larger contributions
# will lead to larger and larger decrease in performance (e.g., larger RMSEs for
# regression).
# This process is repeated multiple times, and the average performance drop is
# used as the importance score, providing a robust measure of each variable's
# contribution.

vi_perm = knn_xplnr.model_parts(
    B=10,  # Number of permutations
    variables=None,  # Specify to only compute for some
)
vi_perm.plot()
# - The vertical line is the baseline RMSE
# - The horizontal bars are the increase in RMSE

vi_perm.plot(max_vars=100)


### Understand a variables contribution ---------

# Partial dependence plots (PDP) visualize the *marginal* effect of one or more
# predictor on the predicted outcome. That is, how the prediction is affected by
# changing the value of variable X while all other *are held constant*
# (_ceteris-paribus_). See:
#
# https://ema.drwhy.ai/partialDependenceProfiles.html
# https://marginaleffects.com/chapters/ml.html

pdp_years = knn_xplnr.model_profile(
    variables=["Years"],
    # default is to plot results of 100 randomly sampled
    # observations.
    N=None,
)
pdp_years.plot()  # average
# Note that this is a KNN model - it has no structure, and yet, this plot is
# fairly easy to understand!
pdp_years.plot(geom="profiles")  # each line is a single outcome

# Division
knn_xplnr.model_profile(
    variables="Division", variable_type="categorical"
).plot()

# Years by Division
knn_xplnr.model_profile(variables="Years", groups="Division").plot()
# Note that we don't have standard errors or confidence intervals.
# Just pure predictions - so these must be taken with a grain of salt.

# For more complex PDP plots using sklearn directly, see also:
help(PartialDependenceDisplay)


# Classification ------------------------------------------------------------
# Let's apply all these methods to a (multi-class, probabilistic) classification
# model.

## Fit a model -------------------------------------
# Let's fit a classification model!

penguins = load_penguins()
penguins.species = penguins.species.astype("category")

outcome = "species"
features = penguins.columns.difference([outcome, "island"])

X_train, X_test, y_train, y_test = train_test_split(
    penguins[features], penguins[outcome], train_size=0.7, random_state=20260125
)


# Preprocessing
s_numeric_features = make_column_selector(dtype_include=np.number)
s_categorical_features = make_column_selector(
    dtype_include=[pd.Categorical, object]
)

# Imputers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), s_numeric_features),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "encode",
                        OneHotEncoder(sparse_output=False),
                    ),
                ]
            ),
            s_categorical_features,
        ),
    ]
)

# We'll fit a random forest model:
rf_spec = RandomForestClassifier(
    max_features="sqrt", n_estimators=500, min_samples_leaf=5
)

rf_pipe = Pipeline([("preprocessor", preprocessor), ("model", rf_spec)])
rf_pipe.fit(X_train, y_train)


## Explain the model -----------------------------------------------

# dalex does not support multiclass classification, so we need to look at each
# class separately. https://github.com/ModelOriented/DALEX/issues/338


# define function that extracts probability of a Adelie class
def predict_Adelie_proba(pipeline: Pipeline, X: pd.DataFrame):
    proba = pipeline.predict_proba(X)
    class_index = list(pipeline.classes_).index("Adelie")
    return proba[:, class_index]


rf_Adelie_xplnr = dx.Explainer(
    rf_pipe,
    label="Random Forest (Adelie)",
    predict_function=predict_Adelie_proba,  # custom predict function
    data=X_train,
    y=y_train == "Adelie",  # boolean outcome for Adelie
)


### Explain a single prediction ------------------------------
# Why does the model think that obs 61 has a high chance of being a Adelie?
print(rf_pipe.predict_proba(X_test.iloc[[61]]))
print(rf_pipe.predict(X_test.iloc[[61]]))


# We can look at his SHAP values:
shap_61 = rf_Adelie_xplnr.predict_parts(X_test.iloc[[61]], type="shap")
shap_61.plot()


### Variable importance ----------------------

# Since we're using a random forest model, we can get model-based variable
# importance:
importances = rf_pipe.named_steps["model"].feature_importances_
feature_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
sorted_idx = importances.argsort()

plt.figure()
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.xlabel("Importance")


# But we can still use the permutation method:
vi_perm = rf_Adelie_xplnr.model_parts(B=10, variable_groups=None)
vi_perm.plot(max_vars=100)
# This matches the plot above very well!
# What's going on with year / island?

### Understand a variables contribution ---------

rf_Adelie_xplnr.model_profile(
    variables=["bill_length_mm", "body_mass_g"]
).plot()
# As far as bill length goes, it seems like smaller bills predict Adelie...

# Let's see if this makes sense...
(
    ggplot(X_train, aes("bill_length_mm", "body_mass_g", color=y_train))
    + geom_point()
)
