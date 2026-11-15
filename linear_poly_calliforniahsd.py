# ============================================================
# LINEAR REGRESSION AND POLYNOMIAL REGRESSION
# CALIFORNIA HOUSING DATASET
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------------------------------------------
# STEP 1: Load California Housing Dataset
# ------------------------------------------------------------

print("Loading California Housing Dataset...")

data = fetch_california_housing(as_frame=True)

df = data.frame

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# STEP 2: Separate Features and Target
# ------------------------------------------------------------

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# ------------------------------------------------------------
# STEP 3: Split Dataset into Training and Testing
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# STEP 4: LINEAR REGRESSION
# ============================================================

print("\nTraining Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)


# Calculate Linear Regression metrics

linear_mse = mean_squared_error(y_test, linear_prediction)
linear_mae = mean_absolute_error(y_test, linear_prediction)
linear_r2 = r2_score(y_test, linear_prediction)


# ============================================================
# STEP 5: POLYNOMIAL REGRESSION
# ============================================================

print("Training Polynomial Regression...")

polynomial_model = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])

polynomial_model.fit(X_train, y_train)

polynomial_prediction = polynomial_model.predict(X_test)


# Calculate Polynomial Regression metrics

polynomial_mse = mean_squared_error(y_test, polynomial_prediction)
polynomial_mae = mean_absolute_error(y_test, polynomial_prediction)
polynomial_r2 = r2_score(y_test, polynomial_prediction)


# ============================================================
# STEP 6: DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("                     MODEL PERFORMANCE")
print("=" * 60)

print("\nLinear Regression:")
print("-" * 30)
print("MSE      :", round(linear_mse, 4))
print("MAE      :", round(linear_mae, 4))
print("R2 Score :", round(linear_r2, 4))


print("\nPolynomial Regression:")
print("-" * 30)
print("MSE      :", round(polynomial_mse, 4))
print("MAE      :", round(polynomial_mae, 4))
print("R2 Score :", round(polynomial_r2, 4))


# ============================================================
# STEP 7: COMPARISON TABLE
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression"
    ],
    "MSE": [
        linear_mse,
        polynomial_mse
    ],
    "MAE": [
        linear_mae,
        polynomial_mae
    ],
    "R2 Score": [
        linear_r2,
        polynomial_r2
    ]
})

print("\n")
print("=" * 60)
print("                    COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# ============================================================
# STEP 8: SAVE RESULTS TO CSV
# ============================================================

results.to_csv("regression_results.csv", index=False)

print("\nResults saved to: regression_results.csv")


# ============================================================
# STEP 9: GRAPH - ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(12, 5))

# Linear Regression graph
plt.subplot(1, 2, 1)

plt.scatter(
    y_test,
    linear_prediction,
    alpha=0.3,
    color="blue"
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Linear Regression")
plt.grid(True)


# Polynomial Regression graph
plt.subplot(1, 2, 2)

plt.scatter(
    y_test,
    polynomial_prediction,
    alpha=0.3,
    color="red"
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Polynomial Regression")
plt.grid(True)

plt.tight_layout()

plt.savefig("regression_comparison.png")

plt.show()


# ============================================================
# STEP 10: FINAL CONCLUSION
# ============================================================

print("\n")
print("=" * 60)
print("                     CONCLUSION")
print("=" * 60)

if polynomial_r2 > linear_r2:
    print("Polynomial Regression has a higher R2 Score.")
else:
    print("Linear Regression has a higher R2 Score.")

if polynomial_mse < linear_mse:
    print("Polynomial Regression has a lower MSE.")
else:
    print("Linear Regression has a lower MSE.")

if polynomial_mae < linear_mae:
    print("Polynomial Regression has a lower MAE.")
else:
    print("Linear Regression has a lower MAE.")