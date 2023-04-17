import numpy as np

class KMeansr:
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.cluster_centers_ = None
        self.labels = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize cluster_centers_ randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update cluster_centers_
            new_cluster_centers_ = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_cluster_centers_[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_cluster_centers_ - self.cluster_centers_)) < self.tol:
                break
                
            self.cluster_centers_ = new_cluster_centers_
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
        
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.cluster_centers_):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances


# # probamos el dataset
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler

# # Specifying the number of cluster our data should have
# n_components = 4

# X, true_labels = make_blobs(
#     n_samples=750, centers=n_components, cluster_std=0.4, random_state=0
# )

# plt.title("Unclustered Data")
# plt.scatter(X[:, 0], X[:, 1], s=15)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# # Initialize KMeans
# kmeans = KMeansr(n_clusters=4)

# # fit the data & predict cluster labels
# kmeans.fit(X)
# predicted_labels = kmeans.predict(X)

# # Based on predicted_labels, we assign each data point distinct colour
# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
# for k, col in enumerate(colors):
#     cluster_data = predicted_labels == k
#     plt.scatter(X[cluster_data, 0], X[cluster_data, 1], s=15)

# plt.title("Clustered Data")
# plt.xticks([])
# plt.yticks([])
# plt.show()