import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics as s

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Load dataset (Make sure heart.csv is in /home/avcoe/ or update path)
df = pd.read_csv("/home/avcoe/heart.csv")
print("First 5 rows:")
print(df.head())

# Information of the dataset
print("\nDataset Info:")
df.info()

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

# Data type count of each column
print("\nData Type Counts:\n", df.dtypes.value_counts())

print("\nStatistical Summary (describe):")
print(df.describe())

# Missing values check
n = df.columns[df.dtypes != 'object']

# DataFrame size and shape
print("Size of dataset is:", df.size)
print("Shape of dataset is:", df.shape)

print("\nMissing Values per Column:")
print(df.isnull().sum())

# Fill missing Age values with median (if any)
if 'age' in df.columns:
    df['age'] = df['age'].fillna(df['age'].median())
    print("\nMissing Values After Handling Age:")
    print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)

# --------------------------------------------------
# Statistical Analysis
# --------------------------------------------------
print("\nMean (Numeric Only):")
print(df.mean(numeric_only=True))

print("\nMedian (Numeric Only):")
print(df.median(numeric_only=True))

print("\nStandard Deviation (Numeric Only):")
print(df.std(numeric_only=True))

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# --------------------------------------------------
# Data Visualization
# --------------------------------------------------
# Histogram of Age
plt.figure(figsize=(8, 5))
plt.hist(df['age'], bins=20, edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram - Cholesterol
if 'chol' in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df['chol'], bins=15, edgecolor='black')
    plt.title('Histogram of Cholesterol')
    plt.xlabel('Cholesterol')
    plt.ylabel('Frequency')
    plt.show()

    # Boxplot - Cholesterol
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['chol'])
    plt.title('Boxplot of Cholesterol')
    plt.show()

# Boxplot - Age
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['age'])
plt.title('Boxplot of Age')
plt.show()

# Average Age of the Patients using statistics module
average_age = s.mean(df['age'])
print("Average age of patients: ", average_age)

# Extracting Particular Columns
print("\nSample Age Data:")
print(df['age'].head())
print("\nSample Sex Data:")
print(df['sex'].head())

# --------------------------------------------------
# Splitting Data into Training and Testing Sets
# --------------------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)