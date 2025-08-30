import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline  # we need a different pipe!
from sklearn import set_config

set_config(display="diagram")

from ISLP import load_data


# The Data & Problem --------------------------------------------------------

Caravan = load_data("Caravan")
print(Caravan.info())


# Data Splitting
outcome = "Purchase"
features = Caravan.columns.difference(["Purchase"])  # all but the outcome

X_train, X_test, y_train, y_test = train_test_split(
    Caravan[features],
    Caravan[outcome],
    train_size=0.7,
    random_state=1234,
    stratify=Caravan[outcome],
)


print(y_train.value_counts(normalize=True))
# As we can see, the classes are very unbalanced.

# Note we used a stratified split, so that both the train and test set have
# about the ~same distribution of classes. This is particularly important with
# imbalanced data.
print(y_test.value_counts(normalize=True))


# The worst model ---------------------------------------------------------
# The data imbalance means that, technically, we can achieve high accuracy by
# simply predicting "No"....

y_pred_BAD = np.array(["No"] * len(y_test))
y_pred_probYes_BAD = np.array([0] * len(y_test))

print(confusion_matrix(y_test, y_pred_BAD))

acc = accuracy_score(y_test, y_pred_BAD)
print(f"Accuracy = {acc:.3f}")

# But as we can see, we have no sensitivity!
sens = recall_score(y_test, y_pred_BAD, pos_label="Yes")
print(f"Sensitivity = {sens:.3f}")


# Training with Imbalance Data --------------------------------------------

knn_pipe = Pipeline(
    steps=[
        ("z", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=10)),
    ]
)

knn_pipe.fit(X_train, y_train)


# Up- and Down-Sampling ---------------------------------------------------

# We can also sample our data such that we artificially achieve class balances.
# The main methods are:
# down-sampling:
#   randomly subset all the classes in the training set so that their class
#   frequencies match the least prevalent class.
# up-sampling:
#   randomly sample (with replacement) the minority class(es) to be the same
#   size as the majority class.
# hybrid methods:
#   techniques such as SMOTE and ROSE down-sample the majority class and
#   synthesize new data points in the minority class.
# See https://imbalanced-learn.org/stable/

# We will use up- and down-sampling:

knn_down_pipe = Pipeline(
    steps=[
        ("downsamp", RandomUnderSampler()),  # added here
        ("z", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=10)),
    ]
)

knn_up_pipe = Pipeline(
    steps=[
        ("upsamp", RandomOverSampler()),  # added here
        ("z", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=10)),
    ]
)

# Note these are imblearn.pipeline.Pipeline() and NOT
# sklearn.pipeline.Pipeline()!

knn_down_pipe.fit(X_train, y_train)
knn_up_pipe.fit(X_train, y_train)


# Comparing Results -------------------------------------------------------

mod_ids = ["FIXED", "None", "Up", "Down"]

# y_pred_BAD
y_pred = knn_pipe.predict(X_test)
y_pred_down = knn_down_pipe.predict(X_test)
y_pred_up = knn_up_pipe.predict(X_test)

accs = []
accs.append(accuracy_score(y_test, y_pred_BAD))
accs.append(accuracy_score(y_test, y_pred))
accs.append(accuracy_score(y_test, y_pred_down))
accs.append(accuracy_score(y_test, y_pred_up))

senss = []
senss.append(recall_score(y_test, y_pred_BAD, pos_label="Yes"))
senss.append(recall_score(y_test, y_pred, pos_label="Yes"))
senss.append(recall_score(y_test, y_pred_down, pos_label="Yes"))
senss.append(recall_score(y_test, y_pred_up, pos_label="Yes"))

for modi, acci in zip(mod_ids, accs):
    print(f"Accuracy ({modi}) = {acci:.3f}")

for modi, sensi in zip(mod_ids, senss):
    print(f"Sensitivity ({modi}) = {sensi:.3f}")

# As we can see, the accuracy has dropped, but sensitivity is higher.
# (What about specificity?)


# We can also compare ROC curves and AUCs:
Y_pred_probYes = {
    "FIXED": y_pred_probYes_BAD,
    "None": knn_pipe.predict_proba(X_test)[:, 1],
    "Up": knn_up_pipe.predict_proba(X_test)[:, 1],
    "Down": knn_down_pipe.predict_proba(X_test)[:, 1],
}

plt.figure()
ax = plt.axes()
for modi in mod_ids:
    RocCurveDisplay.from_predictions(
        y_test,
        Y_pred_probYes[modi],
        pos_label="Yes",
        name=modi,
        ax=ax,
        plot_chance_level=(modi == "Down"),
    )
plt.show()
