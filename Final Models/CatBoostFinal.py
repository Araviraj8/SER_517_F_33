import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Read data
csvpath = 'combined.csv'
data = pd.read_csv(csvpath, engine='python')

# Drop columns with high missing values or irrelevant for classification
columns_to_drop = ['sVid', 'dTos', 'dDSb', 'dTtl', 'dHops', 'SrcGap', 'DstGap',
                   'SrcWin', 'DstWin', 'dVid', 'SrcTCPBase', 'DstTCPBase',
                   'AttackType', 'Attack Tool']
data.drop(columns_to_drop, axis=1, inplace=True)

# Impute missing values
data["sTos"].fillna(0, inplace=True)
data["sDSb"].fillna("cs0", inplace=True)
data["sTtl"].fillna(data["sTtl"].mean(), inplace=True)
data["sHops"].fillna(data["sHops"].mean(), inplace=True)

# Encode the target variable 'Label'
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])  # 1 is malicious, 0 is benign now

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=['Proto', 'sDSb', 'Cause', 'State'], dtype='int')

# Split data into features and target
X = data.drop('Label', axis=1)
y = data['Label']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost classifier
clf = CatBoostClassifier(iterations=1000,  # Number of trees (boosting iterations)
                         learning_rate=0.1,  # Learning rate (shrinkage)
                         depth=6,  # Depth of each tree
                         loss_function='MultiClass',  # Loss function for classification
                         random_seed=42)  # Random seed for reproducibility

# Fit the classifier to the training data
clf.fit(X_train, y_train, verbose=100)  # Verbose=100 prints training progress every 100 iterations

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred).ravel()
print("Confusion Matrix:")
print(conf_matrix)
