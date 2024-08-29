# Customer Segmentation Using K-means Clustering

## Introduction

Customer segmentation is a crucial aspect of any business strategy. It involves dividing customers into distinct groups based on their purchasing behavior and demographics. This allows businesses to tailor their marketing efforts, improve customer satisfaction, and increase sales.

**K-means clustering** is one of the most popular unsupervised machine learning algorithms used for customer segmentation. It aims to partition `n` observations into `k` clusters in which each observation belongs to the cluster with the nearest mean. This method is particularly useful when you have a large dataset and want to find hidden patterns within the data.

## Project Overview

In this project, we use K-means clustering to group customers of a retail store based on their purchase history. The dataset contains information about customers including their age, annual income, and spending score.

### Dataset

The dataset used in this project can be found on Kaggle: [Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). It contains the following columns:

- `CustomerID`: Unique identifier for each customer
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Annual Income (k$)`: Annual income of the customer in thousands of dollars
- `Spending Score (1-100)`: Score assigned by the store based on customer behavior and spending

### Project Steps

1. **Data Preprocessing**: 
    - Load the dataset and check for any missing values.
    - Select relevant features for clustering (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`).
    - Normalize the data to ensure that each feature contributes equally to the distance computation.

2. **Finding the Optimal Number of Clusters**:
    - Use the Elbow Method to determine the optimal number of clusters by plotting the inertia for different values of `k`.

3. **Training the K-means Model**:
    - Train the K-means model using the optimal number of clusters determined from the Elbow Method.
    - Assign each customer to a cluster.

4. **Model Evaluation**:
    - Compute the Silhouette Score and Davies-Bouldin Index to evaluate the quality of the clusters.

5. **Cluster Visualization**:
    - Use pairplots and 3D scatter plots to visualize the clusters.

6. **Saving the Model**:
    - Save the trained K-means model for future use.

### Code Implementation

The following sections provide the complete code for each step mentioned above.

#### Data Preprocessing

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "https://raw.githubusercontent.com/VjChoudhary7/Customer-Segmentation-Tutorial-in-python/master/Mall_Customers.csv"
data = pd.read_csv(url)

# Check for missing values
data.dropna(inplace=True)

# Select relevant features for clustering
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# Finding the Optimal Number of Clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Use the elbow method to find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Training the K-means Model
# Based on the elbow method, choose an optimal number of clusters (e.g., 5)
optimal_clusters = 5

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(scaled_features)

# Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Model Evaluation
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Compute the silhouette score
sil_score = silhouette_score(scaled_features, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Compute the Davies-Bouldin index
db_score = davies_bouldin_score(scaled_features, kmeans.labels_)
print(f'Davies-Bouldin Index: {db_score}')

# Cluster Visualization
import seaborn as sns

# Visualize the clusters using pairplot
sns.pairplot(data, hue='Cluster', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
plt.show()

# 3D scatter plot to visualize the clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], 
                c=data['Cluster'], s=50, cmap='viridis')
plt.colorbar(sc)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.show()

# Saving the Model
import joblib

# Save the K-means model
joblib_file = "kmeans_model.pkl"  
joblib.dump(kmeans, joblib_file)
print(f"Model saved as {joblib_file}")

# Load the K-means model
loaded_kmeans = joblib.load(joblib_file)
print("Model loaded successfully")

# Cluster Profiling
# Analyzing the characteristics of each cluster
for i in range(optimal_clusters):
    print(f'Cluster {i}')
    print(data[data['Cluster'] == i].describe())
    print('\n')
```

# Conclusion
This project demonstrates the application of K-means clustering for customer segmentation. By grouping customers based on their purchase history and demographics, businesses can gain valuable insights and make data-driven decisions to improve their marketing strategies and customer satisfaction.

## Key Points

    K-means Clustering: A popular unsupervised machine learning algorithm used for clustering.
    Elbow Method: A technique to determine the optimal number of clusters.
    Silhouette Score & Davies-Bouldin Index: Metrics to evaluate the quality of the clustering.
    Data Visualization: Pairplots and 3D scatter plots to visualize the clusters.
    Model Saving: Using joblib to save and load the trained model.

For further improvements, consider experimenting with other clustering algorithms and feature engineering techniques to enhance the model's performance.


## References

- [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Customer Segmentation Dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [Davies-Bouldin Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
- [Joblib](https://joblib.readthedocs.io/en/latest/)

