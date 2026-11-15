import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
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

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# 3. Define models and parameter grids
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

# 4. Train, evaluate and optimize each model
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

# 5. Display results
df = pd.DataFrame(results)
print(df.to_string(index=False))