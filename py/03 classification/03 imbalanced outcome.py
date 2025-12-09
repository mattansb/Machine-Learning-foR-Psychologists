import pandas as pd
from plotnine import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import set_config

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline  # we need a different pipe!

from ISLP import load_data

set_config(display="diagram")


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
    random_state=20251202,
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

dummy_model = DummyClassifier().fit(X_train, y_train)
# The dummy model is the worst possible model - it does not use any information
# in X, only the distribution of Y in the training set.
# - For regression, it always predicts mean(Y)
# - For classification, it predicts the frequent class and base rate
#   probabilities.
# These models are good for benchmarking.

# But like a "real" model, it can be used to generate predictions on new data:
y_pred_null = dummy_model.predict(X_test)
y_pred_proba_null = dummy_model.predict_proba(X_test)

# same as:
# y_pred_null = np.array(["No"] * len(y_test))
# y_pred_proba_null = np.tile(
#     y_train.value_counts(normalize=True), (len(y_test), 1)
# )

# Confusion matrix
cm_null = confusion_matrix(y_test, y_pred_null, labels=["Yes", "No"])
print("\nNull model confusion matrix:")
print(pd.DataFrame(cm_null, index=["Yes", "No"], columns=["Yes", "No"]))


# But as we can see, we have no sensitivity!
def specificity_score(y_true, y_pred):
    # Specificity = TN / (TN + FP)
    # For binary classification with pos_label, this is recall of negative class
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=["No", "Yes"]
    ).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


accuracy_null = accuracy_score(y_test, y_pred_null)
sensitivity_null = recall_score(y_test, y_pred_null, pos_label="Yes")
specificity_null = specificity_score(y_test, y_pred_null)

print("\n" + "=" * 60)
print("Null Model Performance:")
print("=" * 60)
print(f"Accuracy:    {accuracy_null:.3f}")
print(f"Sensitivity: {sensitivity_null:.3f}")
print(f"Specificity: {specificity_null:.3f}")
print("=" * 60)


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
# See https://imbalanced-learn.org/

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

# Comparing Results -------------------------------------------------------

# Get predictions from all models
y_pred_none = knn_pipe.predict(X_test)
y_pred_proba_none = knn_pipe.predict_proba(X_test)

y_pred_up = knn_up_pipe.predict(X_test)
y_pred_proba_up = knn_up_pipe.predict_proba(X_test)

y_pred_down = knn_down_pipe.predict(X_test)
y_pred_proba_down = knn_down_pipe.predict_proba(X_test)


# Calculate metrics for each method
metrics_by_method = []
for pred_class, method in zip(
    [y_pred_null, y_pred_none, y_pred_up, y_pred_down],
    ["NULL", "None", "Up", "Down"],
):
    metrics_by_method.append(
        {
            "Method": method,
            "Accuracy": accuracy_score(y_test, pred_class),
            "Sensitivity": recall_score(
                y_test,
                pred_class,
                pos_label="Yes",
            ),
            "Specificity": specificity_score(y_test, pred_class),
        }
    )

metrics_df = pd.DataFrame(metrics_by_method)
print("\n" + "=" * 60)
print("Performance Comparison:")
print("=" * 60)
print(metrics_df.to_string(index=False))
print("=" * 60)
# As we can see, the accuracy (and specificity) have dropped, but sensitivity is
# higher.


# We can also compare ROC curves and AUCs:
roc_data = []
for pred_proba, method in zip(
    [y_pred_proba_null, y_pred_proba_none, y_pred_proba_up, y_pred_proba_down],
    ["NULL", "None", "Up", "Down"],
):
    fpr, tpr, _ = roc_curve(y_test, pred_proba[:, 1], pos_label="Yes")
    roc_data.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "Method": method}))

roc_df = pd.concat(roc_data)

p_roc = (
    ggplot(roc_df, aes(x="fpr", y="tpr", color="Method"))
    + geom_line(size=1)
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="gray")
    + labs(
        x="False Positive Rate (1 - Specificity)",
        y="True Positive Rate (Sensitivity)",
        title="ROC Curves by Sampling Method",
        color="Method",
    )
    + theme_minimal()
    + coord_equal()
    + xlim(0, 1)
    + ylim(0, 1)
)
p_roc.draw(show=True)

# Calculate AUC for each method
for pred_proba, method in zip(
    [y_pred_proba_null, y_pred_proba_none, y_pred_proba_up, y_pred_proba_down],
    ["NULL", "None", "Up", "Down"],
):
    auc = roc_auc_score(y_test, pred_proba[:, 1])
    print(f"  {method}: {auc:.3f}")
