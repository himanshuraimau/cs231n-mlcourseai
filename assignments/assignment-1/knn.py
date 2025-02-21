import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter



class KNN_Classifier:

    def __init__(self,distance_metric, k=3):
        self.distance_metric = distance_metric
        self.k = k
    
    def get_distance_metric(self,training_data_point,test_data_point):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(training_data_point,test_data_point)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(training_data_point,test_data_point)
        elif self.distance_metric == 'chebyshev':
            return self.chebyshev_distance(training_data_point,test_data_point)
        else:
            return self.euclidean_distance(training_data_point,test_data_point)
    
    def euclidean_distance(self,training_data_point,test_data_point):
        return np.sqrt(np.sum(np.square(training_data_point - test_data_point)))
    
    def manhattan_distance(self,training_data_point,test_data_point):
        return np.sum(np.abs(training_data_point - test_data_point))
    
    def chebyshev_distance(self,training_data_point,test_data_point):
        return np.max(np.abs(training_data_point - test_data_point))
    
    def fit(self,training_data,training_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        
    
    def predict(self, test_data):
        predictions = []
        for test_point in test_data:
            # Calculate distances between the test point and all training points
            distances = [self.get_distance_metric(train_point, test_point) for train_point in self.training_data]
            # Use argsort to get indices of the k smallest distances
            k_nearest_indices = np.argsort(distances)[:self.k]
            # Get the labels of the k nearest neighbors
            k_nearest_labels = [self.training_labels[i] for i in k_nearest_indices]
            # Predict the label based on the most common class among the k nearest neighbors
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)


# Load CSV
df = pd.read_csv("sample_data.csv")

# Separate features and labels
X = df.drop(columns=["class"]).values  # Feature columns
y = df["class"].values  # Label column

# Convert class labels to numeric (Iris-setosa -> 0, Iris-versicolor -> 1, etc.)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN
knn = KNN_Classifier(distance_metric="euclidean", k=3)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Print results
print("Predictions:", y_pred)
print("Actual Labels:", y_test)