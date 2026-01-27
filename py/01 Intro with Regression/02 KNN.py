import pandas as pd
import numpy as np
from plotnine import *

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PowerTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import set_config

set_config(display="diagram", transform_output="pandas")

from ISLP import load_data

# Data Loading and Initial Preparation ------------------------------------

Auto = load_data("Auto")
# The Auto Dataset contains information about cars.
print(Auto.info())
# For each car, the following vars were recorded:
#  - cylinders
#     Number of cylinders between 4 and 8
# - displacement
#     Engine displacement (cu. inches)
# - horsepower
#     Engine horsepower
# - weight
#     Vehicle weight (lbs.)
# - acceleration
#     Time to accelerate from 0 to 60 mph (sec.)
# - year
#     Model year (modulo 100)
# - origin
#     Origin of car (1. American, 2. European, 3. Japanese)
Auto["origin"] = pd.Categorical(Auto["origin"])

# We're interested in predicting gas consumption: MPG (miles per gallon)

# But this time we won't be using a linear regression model - we're using KNN!

## 1) Split the data ----------------------------------------------

# Let's first list our outcome and predictors:
outcome = "mpg"
features = ["weight", "origin", "horsepower"]

X_train, X_test, y_train, y_test = train_test_split(
    Auto[features], Auto[outcome], train_size=0.7, random_state=1111
)
print(X_train.shape)
print(X_test.shape)


# 2) Specify the model -------------------------------------------

## Model specification ---------------------

# We will be using 5 nearest neighbors:
neigh5 = KNeighborsRegressor(n_neighbors=5)

## Define preprocessor --------------------

# Since KNN identifies neighbors of observations according to their
# **distance**, the scale of the variables matters: large scale -> larger
# distance between the observations on that X.
# So we need to re-scale all variables. And we also need to dummy code our
# factor (origin).

# There are several ways to do this -
# here's one.

ct = ColumnTransformer(
    transformers=[
        ("dummy", OneHotEncoder(drop="first", sparse_output=False), ["origin"]),
        # The Yeo–Johnson transformation (a generalization of the Box-Cox
        # transformation) can be used to make highly skewed variables resemble
        # a more normal-like distribution, typically improving the performance
        # of the model.
        # https://en.wikipedia.org/wiki/Power_transform#Yeo%E2%80%93Johnson_transformation
        # It requires the "training" of a Lambda parameter, which
        # sklearn.preprocessing finds for us.
        (
            "yj",
            PowerTransformer(method="yeo-johnson", standardize=False),
            ["horsepower"],
        ),
    ],
    remainder="passthrough",
)

# CHAIN them all together:
preprocessor = Pipeline(steps=[("coltrans", ct), ("z", StandardScaler())])
# Note that the ORDER OF STEPS - where we put OneHotEncoder() determines if
# the dummies will be z-scored!


# Visualize distribution before and after transformation
hp_original = X_train["horsepower"].values
hp_transformed = preprocessor.fit_transform(X_train)["yj__horsepower"].values

plot_df = pd.DataFrame(
    {
        "Horsepower": np.concatenate([hp_original, hp_transformed]),
        "Type": ["Original"] * len(hp_original)
        + ["Standardized\nYeo-Johnson"] * len(hp_transformed),
    }
)
plot_df["Type"] = pd.Categorical(
    plot_df["Type"], categories=["Original", "Standardized\nYeo-Johnson"]
)

p = (
    ggplot(plot_df, aes(x="Horsepower", fill="Type"))
    + geom_histogram(bins=30, alpha=0.7, show_legend=False)
    + facet_wrap("~Type", scales="free")
    + scale_fill_manual(values=["skyblue", "pink"])
    + theme_minimal()
    + theme(figure_size=(12, 5))
)
p.draw(show=True)

# Finally, we can combine the preprocessor and the regressor a pipeline -
# together they tell us how data *should* be used to fit a model.
knn_pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", neigh5)]
)


## 3) Fitting the model -------------------------------------------

# Fitting the model is as easy as passing some data to the pipeline:
knn_pipe.fit(X_train, y_train)

## 4) Predict and Evaluate the model -------------------------------

# Generate predictions:
y_pred = knn_pipe.predict(X_test)

plot_data = pd.DataFrame({"y_pred": y_pred, "y_test": y_test})
p = (
    ggplot(plot_data, aes(x="y_pred", y="y_test"))
    + geom_point()
    + geom_abline(intercept=0, slope=1, color="red", linetype="dashed", size=1)
    + labs(x="Estimated: $\hat{mpg}$", y="Truth: mpg")
    + theme_minimal()
)
p.draw(show=True)

# Performance metrics
print(f"R-squared: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.3f}")


# Compare this to performance on the training set:
y_pred2 = knn_pipe.predict(X_train)
print(f"R-squared: {r2_score(y_train, y_pred2):.3f}")
print(f"RMSE: {root_mean_squared_error(y_train, y_pred2):.3f}")


# Exercise ---------------------------------------------------------------

# 1. Build a new model - this time use all avilable predictors.
#  - Make sure to properly pre-process the data.

# 2. Fit a KNN model with k=5

# 3. Evaluate the new model on the test set.
#   How does it compare to the k=5 model with 3 predictors?

# 4. Repeat steps 2 and 3 with k=10. What can we expect to happen?
