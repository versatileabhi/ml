import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# 1. Load dataset directly from scikit-learn (No CSV file required!)
data = load_breast_cancer()
X = data.data
y = data.target

# 2. PCA Visualization (PC1 vs PC2)
X_scaled_full = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)

print("Explained Variance Ratio (PC1, PC2):", pca.explained_variance_ratio_)
print("Total Variance Explained by 2 Components:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="coolwarm",
    alpha=0.75,
    edgecolors="k",
    linewidths=0.4
)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
plt.title("PCA: PC1 vs PC2 (Breast Cancer Dataset)")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=list(data.target_names),
    title="Classes"
)
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_pc1_vs_pc2.png")
plt.show()

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# 4. Define models and parameter grids
models = {
    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC())
        ]),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0]
        }
    )
}

results = []

# 5. Train, evaluate and optimize each model
for name, (model, params) in models.items():

    # Before optimization
    model.fit(X_train, y_train)
    pred_before = model.predict(X_test)

    before_accuracy = accuracy_score(y_test, pred_before)
    before_f1 = f1_score(y_test, pred_before)

    # Hyperparameter tuning
    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # After optimization
    best_model = grid.best_estimator_
    pred_after = best_model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy Before (%)": round(
            before_accuracy * 100, 2
        ),
        "F1 Before": round(before_f1, 4),
        "Accuracy After (%)": round(
            accuracy_score(y_test, pred_after) * 100, 2
        ),
        "Precision After": round(
            precision_score(y_test, pred_after), 4
        ),
        "Recall After": round(
            recall_score(y_test, pred_after), 4
        ),
        "F1 After": round(
            f1_score(y_test, pred_after), 4
        ),
        "Best Parameters": str(grid.best_params_)
    })

# 6. Display results
df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))