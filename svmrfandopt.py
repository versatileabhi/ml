from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset (Make sure breast_cancer.csv is in /home/avcoe/ or update path)
data = pd.read_csv(BASE_DIR/"breast_cancer.csv")
t_col="target" if "target" in data.columns else ("diagnosis" if "diagnosis" in data.columns else data.columns[-1])
y_raw = data[t_col]
y = y_raw.map({"M":0,"B":1,"malignant":0,"benign":1}).fillna(y_raw) if y_raw.dtype=="object" else y_raw
data = data.dropna(axis=1, how="all")
data = data.dropna(axis=1, how="all")
t_col = "target" if "target" in data.columns else ("diagnosis" if "diagnosis" in data.columns else data.columns[-1])
y = data[t_col].astype(str).str.strip().map({"M": 0, "B": 1, "1": 1, "0": 0})
if y.isna().any(): y = pd.Series(pd.factorize(data[t_col])[0], index=data.index)
X = data.drop(columns=[t_col, "id"], errors="ignore").select_dtypes(include="number")
X = X.fillna(X.median(numeric_only=True))

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define models and parameter grids
models = {
    "SVM": (
        Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
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
        "Accuracy Before (%)": round(before_accuracy * 100, 2),
        "F1 Before": round(before_f1, 4),
        "Accuracy After (%)": round(accuracy_score(y_test, pred_after) * 100, 2),
        "Precision After": round(precision_score(y_test, pred_after), 4),
        "Recall After": round(recall_score(y_test, pred_after), 4),
        "F1 After": round(f1_score(y_test, pred_after), 4),
        "Best Parameters": str(grid.best_params_)
    })

# 5. Display results
result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))