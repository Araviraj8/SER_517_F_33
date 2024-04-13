# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split


csvpath = 'combined.csv'
data = pd.read_csv(csvpath, engine='python')
data.drop('Label', axis = 1, inplace = True)

data = pd.get_dummies(data, columns = ['Proto', 'sDSb', 'Cause', 'State', 'AttackType'], dtype = 'int')

# X = data.iloc[:, :-1]
# y = data['Label']


# # Define the condition (target values)
# condition = data['AttackType_Benign'] == '1'
# # Create a new DataFrame with rows that meet the condition
# data = data[:1000][condition]
# # Remove the rows from the original DataFrame
# data = data[~condition]

# # Define the condition (target values)
# condition = data['AttackType_Benign'] == '0'
# # Create a new DataFrame with rows that meet the condition
# data_test = data[condition]
# # Remove the rows from the original DataFrame
# data = data[~condition]

# data_test = data.loc[data['AttackType_Benign'] == 0][:2000]

# data = data[data['AttackType_Benign'] == 0]
# data = data.loc[~data['AttackType_Benign'] == 0][:2000]
# df1.loc[~criteria]
# print(data_test[:100])
# data_test.to_csv('data_test1.csv', index=False)
# #AttackType_Benign,AttackType_HTTPFlood,AttackType_ICMPFlood,AttackType_SYNFlood,AttackType_SYNScan,AttackType_SlowrateDoS,AttackType_TCPConnectScan,AttackType_UDPFlood,AttackType_UDPScan


# Specify the criteria to select rows
criteria = (data['AttackType_Benign'] == 0)
data_test = pd.DataFrame()
# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_Benign'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_HTTPFlood'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_HTTPFlood'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_ICMPFlood'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_ICMPFlood'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_SYNFlood'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_SYNFlood'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_SYNScan'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_SYNScan'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_SlowrateDoS'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_SlowrateDoS'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_TCPConnectScan'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_TCPConnectScan'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_UDPFlood'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_UDPFlood'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)

# Specify the criteria to select rows
criteria = (data['AttackType_UDPScan'] == 0)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)


# Specify the criteria to select rows
criteria = (data['AttackType_UDPScan'] == 1)

# Move 1000 rows matching the criteria from df1 to df2
data_test = pd.concat([data_test, data.loc[criteria].head(1000)], ignore_index=True)
# data = data.loc[~criteria].reset_index(drop=True)
data = pd.concat([data[~criteria], data.loc[criteria, :].iloc[1000:]], ignore_index=True)




data_test.to_csv('data_test1.csv', index=False)
data.to_csv('processed_multiclass2.csv', index=False)

X = data.iloc[:, :-9]  # Features (selecting all columns except last 6)
y = data.iloc[:, -9:]  # Labels (selecting last 6 columns)

# print("y label ", y)

# # train/test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

print("xtrain shape", x_train.shape)
print("ytrain shape", y_train.shape)
