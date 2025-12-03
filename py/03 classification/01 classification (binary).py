import numpy as np
import pandas as pd
from plotnine import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn import set_config

from ISLP import load_data

set_config(display="diagram")


# The data and problem ----------------------------------------------------

# Previously, we've used scikit-learn for a regression problem. Today we are
# looking at classification.

# Smarket dataset contains daily percentage returns for the S&P 500 stock index
# between 2001 and 2005 (1,250 days).
Smarket = load_data("Smarket")
print(Smarket.info())
# For each date, the following vars were recorded:
# - Lag1--Lag5 - percentage returns for each of the five previous trading days.
# - Volume - the number of shares traded on the previous day(in billions).
# - Today - the percentage return on the date in question.
# - Direction - whether the market was Up or Down on this date.

# Assume the following classification task on the Smarket data:
# predict Direction (Up/Down) using the features Lag1 and Lag2.
# If we are not sure how Direction is coded we can use:
print(Smarket.Direction.cat.categories)

print(Smarket["Direction"].value_counts())
# The base rate probability:
print(Smarket["Direction"].value_counts(normalize=True))


# Data Splitting (70%):
features = ["Lag1", "Lag2", "Lag3"]
outcome = "Direction"

X_train, X_test, y_train, y_test = train_test_split(
    Smarket[features],
    Smarket[outcome],
    train_size=0.7,
    random_state=20251202,
)


# We'll start by using a parametric method - logistic regression.

# A logistic regression (with scikit-learn) ---------------------------------

## 1) Specify the model -------------------------------------------

# For logistic regression in scikit-learn, we use LogisticRegression:
logit_spec = LogisticRegression(penalty=None)
help(LogisticRegression)


## 2) Feature Preprocessing -------------------------------------------

# Logistic regression does not require any preprocessing for numeric features -
# and we're not creating any new features here. So we can proceed directly to
# creating a simple pipeline.

logit_pipeline = Pipeline(steps=[("classifier", logit_spec)])

print(logit_pipeline)


## 3) Fit the model ---------------------------------------------------

logit_fit = logit_pipeline.fit(X_train, y_train)

# Extract coefficients
classifier = logit_fit.named_steps["classifier"]
coef_df = pd.DataFrame(
    {
        "feature": features,
        "coefficient": classifier.coef_[0],
        "odds_ratio": np.exp(classifier.coef_[0]),
    }
)
print("\nModel coefficients:")
print(coef_df)
print(f"Intercept: {classifier.intercept_[0]:.4f}")
# Or we can...


## 4) Predict and evaluate -------------------------------------------------

# Class predictions (default)
y_pred_class = logit_fit.predict(X_test)
print("\nFirst 10 class predictions:", y_pred_class[:10])

# Probability predictions
y_pred_proba = logit_fit.predict_proba(X_test)
print("\nFirst 10 probability predictions:")
print(y_pred_proba[:10])
# These columns are the probabilistic predictions for each class. They add up to
# 1 for each row.


# Confusion matrix
cm = confusion_matrix(y_test, y_pred_class, labels=["Down", "Up"])
print("\nConfusion matrix:")
print(pd.DataFrame(cm, index=["Down", "Up"], columns=["Down", "Up"]))

# Calculate metrics
# In scikit-learn, we need to specify which class is "positive". By default,
# scikit-learn uses the "1" level, which we do not have in this case - but we
# can override this with the pos_label argument:
accuracy = accuracy_score(y_test, y_pred_class)
sensitivity = recall_score(y_test, y_pred_class, pos_label="Up")
specificity = recall_score(y_test, y_pred_class, pos_label="Down")
f1 = f1_score(y_test, y_pred_class, pos_label="Up")

print("\n" + "=" * 60)
print("Classification Metrics:")
print("=" * 60)
print(f"Accuracy:    {accuracy:.3f}")
print(f"Sensitivity: {sensitivity:.3f}  (recall for 'Up')")
print(f"Specificity: {specificity:.3f}  (recall for 'Down')")
print(f"F1-score:    {f1:.3f}")
print("=" * 60)
# Overall, not amazing...

# Since this is a probabilistic model, we can also look at the ROC curve and AUC:
# For binary classification, we use the probability of the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label="Up")

# Create dataframe for plotting
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

p_roc = (
    ggplot(roc_df, aes(x="fpr", y="tpr"))
    + geom_line(size=1, color="blue")
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="gray")
    + labs(
        x="False Positive Rate (1 - Specificity)",
        y="True Positive Rate (Sensitivity)",
        title="ROC Curve",
    )
    + theme_minimal()
    + coord_equal()
    + xlim(0, 1)
    + ylim(0, 1)
)
p_roc.draw(show=True)

# Or at the Specificity-Sensitivity trade-off:
roc_df["specificity"] = 1 - roc_df["fpr"]
roc_df["sensitivity"] = roc_df["tpr"]

# Reshape for plotting
roc_long = pd.melt(
    roc_df,
    id_vars=["threshold"],
    value_vars=["specificity", "sensitivity"],
    var_name="metric",
    value_name="value",
)

p_threshold = (
    ggplot(roc_long, aes(x="threshold", y="value", color="metric"))
    + geom_line(size=1)
    + labs(
        x="Threshold",
        y="Probability",
        color=None,
        title="Specificity-Sensitivity Trade-off",
    )
    + theme_classic()
    + xlim(0, 1)
    + ylim(0, 1)
)
p_threshold.draw(show=True)
# We hope to see these are not just be mirror images of each other...

# And indeed...
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print(f"\nROC AUC: {auc:.3f}")
