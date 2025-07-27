import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification  # For sample data
from sklearn.model_selection import train_test_split, FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


# The Data & Problem --------------------------------------------------------

X, y = make_classification(
    n_samples=600, class_sep=0.5, weights=[0.75, 0.25], random_state=42
)
# Let us assume y is diagnosis of Alzheimer's disease.


# Data Splitting -
# We will need an additional validation set here - we'll learn about these next
# time!
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.2 / (0.2 + 0.2)), random_state=123
)


# Fit a model -------------------------------------------------------------

logit_pipe = Pipeline(
    steps=[
        ("z", StandardScaler()),
        ("classifer", LogisticRegression(penalty=None)),
    ]
)
logit_pipe.fit(X_train, y_train)


y_pred_val = logit_pipe.predict(X_val)

accuracy_score(y_val, y_pred_val)
recall_score(y_val, y_pred_val)
# The model is pretty good, but early detection of Alzheimer's disease is
# important - we want higher sensitivity (at the cost of lower specificity).
# (Note also that the data is imbalanced, but even so...)

# What can we do?

# Find threshold -----------------------------------------------------------
# We will find the best threshold by using the hold out validation set.

y_pred_prob_val = logit_pipe.predict_proba(X_val)

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob_val[:, 1])


plt.figure()
sns.lineplot(x=thresholds, y=tpr, color="red", label="Sens.")
sns.lineplot(x=thresholds, y=1 - fpr, color="blue", label="Spec.")
sns.lineplot(x=[0.5] * 2, y=[0, 1], color="black", label="Threshold")
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.legend(loc="lower right")
plt.show()
# Looks like a threshold of about 0.2 will give us a high sens without too much
# loss of spec.


## Select threshold -----------------------------------

# We introduce a new type of step in our "pipeline": POST-processing. This is a
# step that adjusts predictions.
# There are many types of adjustments...

# ... we want to adjust the binary classifications threshold.
logit_pipe_02 = FixedThresholdClassifier(logit_pipe, threshold=0.2)

# There are many more adjustments that can be made:
# https://scikit-learn.org/stable/api/sklearn.calibration.html
# https://scikit-learn.org/stable/api/sklearn.model_selection.html#post-fit-model-tuning


## Predict (+ adjust) on test set ---------------------------------

y_pred_05 = logit_pipe.predict(X_test)
y_pred_02 = logit_pipe_02.predict(X_test)

# We can see that the class predictions do not completely agree:
confusion_matrix(y_pred_05, y_pred_02)


accuracy_score(y_test, y_pred_05)
accuracy_score(y_test, y_pred_02)

recall_score(y_test, y_pred_05)
recall_score(y_test, y_pred_02)

# (Note that probabilistic metrics are unaffected by adjustment)
roc_auc_score(y_test, logit_pipe.predict_proba(X_test)[:, 1])
roc_auc_score(y_test, logit_pipe_02.predict_proba(X_test)[:, 1])
