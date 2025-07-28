import matplotlib.pyplot as plt

from ISLP import load_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
)
from sklearn import set_config

set_config(display="diagram")

# The data and problem ----------------------------------------------------

# Previously, we've used scikit learn for a regression problem. Today we are
# looking at classification.

# Smarket dataset contains daily percentage returns for the S&P 500 stock index
# between 2001 and 2005 (1,250 days).
Smarket = load_data("Smarket")
# For each date, the following vars were recorded:
# - Lag1--Lag5 - percentage returns for each of the five previous trading days.
# - Volume - the number of shares traded on the previous day(in billions).
# - Today - the percentage return on the date in question.
# - Direction - whether the market was Up or Down on this date.

# Assume the following classification task on the Smarket data:
# predict Direction (Up/Down) using the features Lag1 and Lag2.
# If we are not sure how Direction is coded we can use levels():
print(Smarket["Direction"].cat.categories)

print(Smarket["Direction"].value_counts())
# The base rate probability:
print(Smarket["Direction"].value_counts(normalize=True))


# Data Splitting (70%):
# Let's first list our outcome and predictors:
outcome = "Direction"
features = ["Lag1", "Lag2", "Lag3"]

X_train, X_test, y_train, y_test = train_test_split(
    Smarket[features], Smarket[outcome], train_size=0.7, random_state=1234
)


# We'll start by using a parametric method - logistic regression.

# A logistic regression (with scikit learn) ---------------------------------

clf = LogisticRegression(penalty=None, fit_intercept=True)
# Note that the default is to apply a penalty - we will talk about penalties in
# a later lesson.

## 2) Feature Preprocessing -------------------------------------------

# Logistic regression does not require any preprocessing.
# But for consistancy...

logit_pipe = Pipeline(
    [
        ("preprocessor", "passthrough"),  # explicitly states: DO NOTHING
        ("classifier", clf),
    ]
)


## 3) Fit the model ---------------------------------------------------

logit_pipe.fit(X_train, y_train)

# Now we can look at the coefficients...
classifier = logit_pipe.named_steps["classifier"]
print(classifier.coef_)

# Or...


## 4) Predict and evaluate -------------------------------------------------

y_pred_class = logit_pipe.predict(X_test)  # predicts classes
print(y_pred_class[0:5])

# But a logistic regreesion is a probabilistic classifier:
y_pred_prob = logit_pipe.predict_proba(X_test)
print(y_pred_prob[0:5, :])
# The columns correspond to the probabilistic predictions for each class.
# They add up to 1 for each row.


print(confusion_matrix(y_test, y_pred_class))


acc = accuracy_score(y_test, y_pred_class)
sens = recall_score(
    y_test, y_pred_class, pos_label="Up"
)  # recall = sensetivity
spec = recall_score(
    y_test, y_pred_class, pos_label="Down"
)  # specificty is the sen for negative class
f1 = fbeta_score(y_test, y_pred_class, pos_label="Up", beta=1)

print(f"Accuracy = {acc:.3f}")
print(f"Sensetivity = {sens:.3f}")
print(f"Specificty = {spec:.3f}")
print(f"F1 = {f1:.3f}")


# Since this is a probabilistic model, we can also look at the ROC curve and AUC:
y_pred_prob_up = y_pred_prob[:, 1]
plt.figure()
RocCurveDisplay.from_predictions(
    y_test, y_pred_prob_up, pos_label="Up", plot_chance_level=True
)
plt.show()

# Or at the Specificity-Sensitivity trade-off:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_up, pos_label="Up")

plt.figure()
plt.plot(thresholds, 1 - fpr, color="red", lw=2, label="Specificty")
plt.plot(thresholds, tpr, color="royalblue", lw=2, label="Sensetivity")
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.legend(loc="lower right")
plt.show()


# And indeed...
auc = roc_auc_score(y_test, y_pred_prob_up)
print(f"AUC = {auc:.3f}")
