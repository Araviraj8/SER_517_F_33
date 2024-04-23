# Import packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder          
from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt 
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Read data
#url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'   
csvpath = 'C:/Users/waghs/Desktop/ser517/Combined.csv'
data = pd.read_csv(csvpath, engine='python')
data.drop('sVid', axis = 1, inplace = True)
data.drop('dTos', axis = 1, inplace = True)
data.drop('dDSb', axis = 1, inplace = True)
data.drop('dTtl', axis = 1, inplace = True)
data.drop('dHops', axis = 1, inplace = True)
data.drop('SrcGap', axis = 1, inplace = True)
data.drop('DstGap', axis = 1, inplace = True)
data.drop('SrcWin', axis = 1, inplace = True)
data.drop('DstWin', axis = 1, inplace = True)
data.drop('dVid', axis = 1, inplace = True)
data.drop('SrcTCPBase', axis = 1, inplace = True)
data.drop('DstTCPBase', axis = 1, inplace = True)
data.drop('Attack Type', axis = 1, inplace = True)
data.drop('Attack Tool', axis = 1, inplace = True)

data["sTos"].fillna(0, inplace=True)
data["sDSb"].fillna("cs0", inplace=True)
data["sTtl"].fillna(data["sTtl"].mean(), inplace=True)
data["sHops"].fillna(data["sHops"].mean(), inplace=True)



# Assuming 'target_column' is the name of your target column in your DataFrame
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])   # 1 is malicious, 0 is benign now

data = pd.get_dummies(data, columns = ['Proto', 'sDSb', 'Cause', 'State'], dtype = 'int') 
# print(one_hot_encoded_data)

print(data['Label'].unique)
# # Check missing values
print(data.isnull().sum())
# # Drop columns with too many missing values
# data.drop('Cabin', axis=1, inplace=True)

# # Drop data with missing values 
# data.dropna(inplace=True)

# Inspect data
data.head()
print(data.shape)
# Transfrom attribute
# le = LabelEncoder()
# data['Sex'] = le.fit_transform(data['Sex'])


# Export DataFrame to CSV file
#data.to_csv('processed.csv', index=False)  # Set index=False if you don't want to export the index


# # Decide variables to use
# X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
# y = data['Survived']
print(data)

X = data.iloc[:, :-1]
y = data['Label']


# # train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=65)






# # Assuming df is your DataFrame with features
# correlation_matrix = data.corr()



# # Plotting the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()





#only accuracy- shiv

model = xgb.XGBClassifier()
model.fit(x_train, y_train)
# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary).ravel()

print("Confusion Matrix:")
print(conf_matrix)


###############

# XGBoost (different learning rate)
# learning_rate_range = np.arange(0.01, 1, 0.05)
# test_XG = [] 
# train_XG = []
# count =0
# for lr in learning_rate_range:
#     xgb_classifier = xgb.XGBClassifier(eta = lr)
#     xgb_classifier.weights = None
#     xgb_classifier.fit(x_train, y_train)
#     #train_XG.append(xgb_classifier.score(x_train, y_train))
#     test_XG.append(xgb_classifier.score(x_test, y_test))
#     count += 1
#     print("count", count)
#     print(test_XG)
# # Line plot
# print("test xg", test_XG)
# print("train", train_XG)
# fig = plt.figure(figsize=(10, 7))
# plt.plot(learning_rate_range, train_XG, c='orange', label='Train')
# plt.plot(learning_rate_range, test_XG, c='m', label='Test')
# plt.xlabel('Learning rate')
# plt.xticks(learning_rate_range)
# plt.ylabel('Accuracy score')
# plt.ylim(0.6, 1)
# plt.legend(prop={'size': 12}, loc=3)
# plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
# plt.show()



















# # Resolve overfitting 
# # new learning rate range

# learning_rate_range = np.arange(0.01, 0.5, 0.05)
# fig = plt.figure(figsize=(19, 17))
# idx = 1
# count = 0
# # grid search for min_child_weight
# for weight in np.arange(0, 4.5, 0.5):
#     train = []
#     test = []
    
#     for lr in learning_rate_range:
#         xgb_classifier = xgb.XGBClassifier(eta = lr, reg_lambda=1, min_child_weight=weight)
#         xgb_classifier.weights = None
#         xgb_classifier.fit(x_train, y_train)
#         train.append(xgb_classifier.score(x_train, y_train))
#         test.append(xgb_classifier.score(x_test, y_test))
#         count += 1
#         print("count", count)
#         print(train)
#     fig.add_subplot(3, 3, idx)
#     idx += 1
#     plt.plot(learning_rate_range, train, c='orange', label='Training')
#     plt.plot(learning_rate_range, test, c='m', label='Testing')
#     plt.xlabel('Learning rate')
#     plt.xticks(learning_rate_range)
#     plt.ylabel('Accuracy score')
#     plt.ylim(0.6, 1)
#     plt.legend(prop={'size': 12}, loc=3)
#     title = "Min child weight:" + str(weight)
#     plt.title(title, size=16)
# plt.show()