# Data Classification Using AI - Project 2
# Decode Labs AI Internship

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Step 1: Load the Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data       # Features: sepal length, sepal width, petal length, petal width
y = iris.target     # Target classes: 0=Setosa, 1=Versicolor, 2=Virginica

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(iris.target_names)} classes")
print(f"Feature names: {iris.feature_names}")
print(f"Class names: {list(iris.target_names)}")
print("-" * 50)

# Step 2: Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data split completed:")
print(f"  Training samples: {X_train.shape[0]} (80%)")
print(f"  Testing samples:  {X_test.shape[0]} (20%)")
print("-" * 50)

# Step 3: Feature scaling using StandardScaler (fit on training data only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling applied using StandardScaler.")
print("-" * 50)

# Step 4: Create and train KNN classifier with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN classifier created with K=5")
print("Model trained on training data.")
print("-" * 50)

# Step 5: Make predictions on test data
y_pred = knn.predict(X_test)
print("Predictions made on test data.")
print("-" * 50)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("=" * 50)
print("           MODEL EVALUATION RESULTS")
print("=" * 50)
print(f"\nAccuracy Score:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"F1 Score:        {f1:.4f}")
print(f"\nConfusion Matrix:")
print(conf_matrix)
print("=" * 50)

# Step 7: Show some sample predictions
print("\nSample Predictions:")
print("-" * 50)
print(f"{'Actual':<15} {'Predicted':<15} {'Result'}")
print("-" * 50)
for i in range(min(10, len(y_test))):
    actual_name = iris.target_names[y_test[i]]
    predicted_name = iris.target_names[y_pred[i]]
    status = "Correct" if y_test[i] == y_pred[i] else "Wrong"
    print(f"{actual_name:<15} {predicted_name:<15} {status}")

print("-" * 50)
print(f"\nTotal correct: {sum(y_test == y_pred)}/{len(y_test)}")
print("Classification complete!")
