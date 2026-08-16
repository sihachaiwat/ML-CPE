import numpy as np

class TFKMeans:
    def __init__(self, n_clusters=4, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X):
        np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Random select centroids begin from data
        initial_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[initial_indices]

        for i in range(1, self.max_iter + 1):
            self.n_iter_ = i
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            
            labels = np.argmin(distances, axis=1)

            # Update Centroids 
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
                for k in range(self.n_clusters)
            ])

            # Check stable
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        self.cluster_centers_ = centroids
        self.labels_ = labels

        # Calculate Inertia (Sum of squared distances to nearest centroid)
        min_distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        self.inertia_ = float(np.sum(min_distances ** 2))

        return self

    def fit_predict(self, X):
        return self.fit(X).labels_