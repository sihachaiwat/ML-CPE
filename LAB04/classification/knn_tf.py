import numpy as np

class TFKNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        
        # Distance
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority Vote
            counts = np.bincount(k_nearest_labels)
            predictions.append(np.argmax(counts))
            
        return np.array(predictions)

    def score(self, X_val, y_val): #calculate acc 
        y_pred = self.predict(X_val)
        return float(np.mean(y_pred == y_val))