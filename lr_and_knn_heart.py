# ============================================================
# EXPERIMENT 7
# Classification using Logistic Regression and KNN
# ============================================================

# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# Step 2: Load the dataset
# ------------------------------------------------------------

df = pd.read_csv("heart.csv")

print("First 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# Step 3: Check missing values
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# Step 4: Separate features and target
# ------------------------------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ------------------------------------------------------------
# Step 5: Split data into training and testing
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ------------------------------------------------------------
# Step 6: Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")


# Create model
lr_model = LogisticRegression(max_iter=1000)

# Train model
lr_model.fit(X_train_scaled, y_train)

# Prediction
y_pred_lr = lr_model.predict(X_test_scaled)


# ------------------------------------------------------------
# Evaluation of Logistic Regression
# ------------------------------------------------------------

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)

print("\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)


# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr
)

plt.title("Logistic Regression - Confusion Matrix")
plt.show()


# ============================================================
# MODEL 2: K-NEAREST NEIGHBORS
# ============================================================

print("\n==============================")
print("K-NEAREST NEIGHBORS (KNN)")
print("==============================")


# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train model
knn_model.fit(X_train_scaled, y_train)

# Prediction
y_pred_knn = knn_model.predict(X_test_scaled)


# ------------------------------------------------------------
# Evaluation of KNN
# ------------------------------------------------------------

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)

print("Accuracy :", knn_accuracy)
print("Precision:", knn_precision)
print("Recall   :", knn_recall)
print("F1 Score :", knn_f1)

print("\nConfusion Matrix:")
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)


# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_knn
)

plt.title("KNN - Confusion Matrix")
plt.show()


# ============================================================
# COMPARISON OF BOTH MODELS
# ============================================================

print("\n======================================")
print("COMPARISON OF BOTH MODELS")
print("======================================")

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Logistic Regression": [
        lr_accuracy,
        lr_precision,
        lr_recall,
        lr_f1
    ],

    "KNN": [
        knn_accuracy,
        knn_precision,
        knn_recall,
        knn_f1
    ]
})

print(comparison)