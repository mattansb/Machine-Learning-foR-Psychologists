import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import patsy
import statsmodels.formula.api as smf
from scipy.stats import zscore  # noqa

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

print(Auto.head())
# We're interested in predicting gas consumption: MPG (miles per gallon).


## Linear Regression (using statsmodels) ----------------------------------
# We'll start with an example of fitting and evaluating a regression model using
# statsmodels. Specifically, we'll fit a *linear regression* model (remember
# that in ML-speak, "regression" is any prediction model with a quantitative
# outcome.)


## 1) Split the data ----------------------------------------------
# We will TRAIN the model (i.e. fit) on the 70% of the observations randomly
# assigned and TEST the model (i.e. predict and assess performance) on the 30%
# that were left.

# because we will use random sampling we need to set a random seed in order to
# replicate the results
np.random.seed(1111)

i = np.random.choice(
    Auto.shape[0], size=int(0.7 * Auto.shape[0]), replace=False
)
Auto_train = Auto.iloc[i]
Auto_test = Auto.iloc[[ix for ix in range(len(Auto)) if ix not in i]]  # ew


## 2) Specify the model and Preprocessing ---------------------------
#  i. What is the outcome? What are the predictors?
#     Does anything need to be transformed or preprocessed somehow?
# ii. How will the predictors be used to predict the outcome?

# In statsmodels.formula.api, step i is typically done with a formula:
f = "mpg ~ origin + zscore(weight) * horsepower"

# Outcome: mpg
# Predictors: origin, weight, horsepower
# Preprocessing:
# - origin is a factor, so will produce dummy coding
# - weight is standardized
# - adding an interaction between (standardized) weight and horsepower

# We can see that all this happens by using the patsy.dmatrices() method:
y_mf, X_mf = patsy.dmatrices(f, Auto_train, return_type="dataframe")
print(X_mf.head())


# The manner the predictors will be used to predict the outcome is determined by
# the fitting function used. Here, we want a linear regression, so we will use
# the lm() function:
help(smf.ols)


## 3) Fitting the model -------------------------------------------
# Fitting, or statistical learning, or training is the process of finding the
# best-fitting model for the data. In the case of linear regression, this means
# finding the coefficients that minimize the sum of squared errors.

# We combine the formula, and data and the fitting function as defined above:
model = smf.ols(
    "mpg ~ origin + zscore(weight) * horsepower", data=Auto_train
).fit()


## 4) Evaluate the model ------------------------------------------
# After fitting the model to the training data set, we can see how well it
# performs on the test set.

# Generate predictions on the test set:
Auto_test.mpg_pred = model.predict(Auto_test)


# Plot estimated values vs truth
plt.figure()
sns.scatterplot(x=Auto_test.mpg_pred, y=Auto_test.mpg)
plt.plot(
    [Auto_test.mpg.min(), Auto_test.mpg.max()],
    [Auto_test.mpg.min(), Auto_test.mpg.max()],
    color="red",
    linestyle="--",
    lw=2,
    label="Identity Line",
)
plt.xlabel("Estimated: $\hat{mpg}$")
plt.ylabel("Truth: mpg")
plt.show()


# How we assess model performance?
# For regression problems- R-squared, MSE, RMSE, MAE...
r2_model = np.corrcoef(Auto_test.mpg, Auto_test.mpg_pred)[0, 1] ** 2
rmse_model = np.sqrt(np.mean((Auto_test.mpg - Auto_test.mpg_pred) ** 2))

print(f"R-squared: {r2_model:.3f}")
print(f"RMSE: {rmse_model:.3f}")


# Let's do this again, with {Scikit-learn}.


# (Linear) Regression with {scikit-learn} -------------------------------------

# We will be using the scikit-learn (sklearn) library.
# (See: https://scikit-learn.org/stable/index.html)

# Let's load the functions and methods we need:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import set_config

set_config(display="diagram")

# It also provides various tools for model fitting, data preprocessing, model
# selection, model evaluation, and many other utilities...


## 1) Split the data ----------------------------------------------

# Let's first list our outcome and predictors:
outcome = "mpg"
features = ["weight", "origin", "horsepower"]

X_train, X_test, y_train, y_test = train_test_split(
    Auto[features], Auto[outcome], train_size=0.7, random_state=1111
)
print(X_train.shape)
print(X_test.shape)


## 2) Specify the model and Preprocessing ---------------------------------

### i. Preprocessing ---------------------------------

# We will create a few preprocessors using ColumnTransformer() and Pipeline().
# - ColumnTransformer() builds a set of instructions for how to treat specific
#   columns. These instructions - transformers - are run in parrallel, and each
#   can be trained with a fit() method and then be applied with a transform()
#   method.
# - Pipeline() builds a set of step that will be applied to data *sequentially*
#   to data. Each step is trained with a fit() method and the applied with a
#   transform method().

ct = ColumnTransformer(
    transformers=[
        # - origin is a factor, so we'll produce dummy coding
        ("dummy", OneHotEncoder(drop="first", sparse_output=False), ["origin"]),
        # - weight is standardized
        ("z", StandardScaler(), ["weight"]),
    ],
    remainder="passthrough",
).set_output(transform="pandas")


def f_interact(x):
    """
    This function takes a data frame, then adds a column
    that is the product of and remainder__horsepower
    """
    x["int"] = x["z__weight"] * x["remainder__horsepower"]
    return x


# CHAIN them all together:
preprocessor = Pipeline(
    steps=[("coltrans", ct), ("int", FunctionTransformer(f_interact))]
)

# This is quite verbose compared to the formula interface, but will come in
# handy with more complicated preprocessing steps.
print(preprocessor)

# Right now, the transformer is just a list of general instructions. To get a
# transformer with specific instructions, we need to train the transformer.
preprocessor.fit(X_train, y_train)


# We can then pre-process any data we want:
preprocessor.transform(X_train)
# Compare this to the model matrix above


### ii. Define the type of model ---------------------

# We want a linear regression model:
reg = LinearRegression(fit_intercept=True)

# Specifying the model type is seperate then fitting the model itself (i.e.,
# finding its paramseters).


# Finally, we can combine the preprocessor and the model type to a pipeline -
# together they tell us how data *should* be used to fit a model.
linreg_pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", reg)]
)

print(linreg_pipe)


## 3) Fitting the model -------------------------------------------

# Fitting the model is as easy as passing training data to the fit() method:
linreg_pipe.fit(X_train, y_train)

# We can extract the underlying model object:
regressor = linreg_pipe.named_steps["regressor"]

pd.DataFrame(
    {
        "statsmodels": model.params,
        "sklearn": np.concatenate(
            (regressor.intercept_.reshape(1, 1), regressor.coef_.reshape(1, 5)),
            axis=1,
        ).flatten(),
    }
)
# (Why aren't these exactly the same? How is this related to bias or variance?)


### 4) Evaluate the model ------------------------------------------

# Generate predictions on the prepared test data.
y_pred = linreg_pipe.predict(X_test)
# The test set is preprocessed according to the pipeline, and then predictions
# are made.

# Plot estimated values vs truth
plt.figure()
sns.scatterplot(x=y_pred, y=y_test)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    lw=2,
    label="Identity Line",
)
plt.xlabel("Estimated: $\hat{mpg}$")
plt.ylabel("Truth: mpg")
plt.show()


# Assess model performance using R-squared and RMSE:
print(f"R-squared: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.3f}")
