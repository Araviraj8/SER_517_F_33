import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Read data
#url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'   
csvpath = 'C:/Users/vchavhan/Desktop/project/Combined.csv'
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

le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])   # 1 is malicious, 0 is benign now

data = pd.get_dummies(data, columns = ['Proto', 'sDSb', 'Cause', 'State'], dtype = 'int')
# print(one_hot_encoded_data)

print(data['Label'].unique)
# # Check missing values
print(data.isnull().sum())
# # Drop columns with too many missing values
# data.drop('Cabin', axis=1, inplace=True)