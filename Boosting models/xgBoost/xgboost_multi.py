import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data into DMatrix format (optimized data structure for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters for XGBoost
param = {
    'max_depth': 3,  # Maximum depth of a tree
    'eta': 0.3,  # Learning rate
    'objective': 'multi:softmax',  # Objective function for multi-class classification
    'num_class': 3  # Number of classes in the dataset
}

# Train the model
num_round = 10  # Number of boosting rounds
model = xgb.train(param, dtrain, num_round)

# Make predictions on the test set
predictions = model.predict(dtest)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
