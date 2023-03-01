from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

# Extract the centroid feature
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)
X_train_centroid = kmeans.transform(X_train)
X_test_centroid = kmeans.transform(X_test)

# Train a KNN classifier on the centroid feature
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_centroid, y_train)

# Evaluate the classifier
accuracy = knn.score(X_test_centroid, y_test)
print(f"Accuracy: {accuracy}")
