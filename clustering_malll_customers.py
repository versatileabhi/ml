# ============================================================
# Practical 8: Clustering Techniques
# K-Means and Hierarchical Clustering
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

data = pd.read_csv("Mall_Customers.csv")

print("First 5 records:")
print(data.head())

# ------------------------------------------------------------
# 2. Select Features
# ------------------------------------------------------------

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features:")
print(X.head())

# ------------------------------------------------------------
# 3. Elbow Method
# ------------------------------------------------------------

wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# ------------------------------------------------------------
# 4. K-Means Clustering (Assuming optimal K = 5)
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

data['KMeans_Cluster'] = kmeans.fit_predict(X)

print("\nK-Means Cluster Results:")
print(
    data[
        ['CustomerID',
         'Annual Income (k$)',
         'Spending Score (1-100)',
         'KMeans_Cluster']
    ].head(20)
)

# ------------------------------------------------------------
# 5. Visualize K-Means Clusters
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=data['KMeans_Cluster'],
    s=50,
    cmap='viridis'
)

# Plot centroids
centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=200,
    c='red',
    label='Centroids'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 6. Hierarchical Clustering - Dendrogram
# ------------------------------------------------------------

linked = linkage(
    X,
    method='ward'
)

plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# ------------------------------------------------------------
# 7. Agglomerative Hierarchical Clustering
# ------------------------------------------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=5,
    metric='euclidean',
    linkage='ward'
)

data['Hierarchical_Cluster'] = hierarchical.fit_predict(X)

print("\nHierarchical Clustering Results:")
print(
    data[
        ['CustomerID',
         'Annual Income (k$)',
         'Spending Score (1-100)',
         'Hierarchical_Cluster']
    ].head(20)
)

# ------------------------------------------------------------
# 8. Visualize Hierarchical Clusters
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=data['Hierarchical_Cluster'],
    s=50,
    cmap='plasma'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Hierarchical Clustering")
plt.show()

# ------------------------------------------------------------
# 9. Display Cluster Counts
# ------------------------------------------------------------

print("\nK-Means Cluster Counts:")
print(data['KMeans_Cluster'].value_counts())

print("\nHierarchical Cluster Counts:")
print(data['Hierarchical_Cluster'].value_counts())