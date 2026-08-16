import numpy as np

class KNNClusterAssigner:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype=np.float64)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test, dtype=np.float64)
        predictions = []

        for x in X_test:
            # Euclidean Distance 
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Find nearest k
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            
            # Majority Vote
            counts = np.bincount(nearest_labels)
            predictions.append(np.argmax(counts))

        return np.array(predictions)