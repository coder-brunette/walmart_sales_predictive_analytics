# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Load train dataset
train_path = 'archive/train.csv'  
train_data = pd.read_csv(train_path)

# Load test dataset
test_path = 'archive/test.csv' 
test_data = pd.read_csv(test_path)

# Load features dataset
features_path = 'archive/features.csv'  
features_data = pd.read_csv(features_path)

# Load store dataset
store_path = 'archive/stores.csv'  
store_data = pd.read_csv(store_path)

# The shape of the dataset
# print(train_data.shape)
# print(test_data.shape)
# print(features_data.shape)
# print(store_data.shape)

# The statistical summary of the dataset
# print(train_data.describe())
# print(test_data.describe())
# print(features_data.describe())
# print(store_data.describe())

# The data type of each column of the dataset
# print(train_data.dtypes)
# print(test_data.dtypes)
# print(features_data.dtypes)
# print(store_data.dtypes)

# Display the first few rows of each dataset
# print("Train Data:")
# print(train_data.head())

# print("\nTest Data:")
# print(test_data.head())

# print("\nFeatures Data:")
# print(features_data.head())

# print("\nStore Data:")
# print(store_data.head())

# Merge train and features data
train_data = pd.merge(train_data, features_data, on=['Store', 'Date', 'IsHoliday'], how='left')

# Merge test and features data
test_data = pd.merge(test_data, features_data, on=['Store', 'Date', 'IsHoliday'], how='left')

# Display the first few rows after merging
# print("Merged Train Data:")
# print(train_data.head())

# print("\nMerged Test Data:")
# print(test_data.head())

# # Check for the missing values
# print("\nMissing Values in Merged Train Data:")
# print(train_data.isnull().sum())

# print("\nMissing Values in Merged Train Data:")
# print(test_data.isnull().sum())

# Handling missing values
# For simplicity, let's fill missing numeric values with their mean
train_data.fillna(train_data.mean(), inplace=True)

test_data.fillna(test_data.mean(), inplace=True)

# # Check again for missing values
# print("\nMissing Values in Merged Train Data after handling:")
# print(train_data.isnull().sum())

# print("\nMissing Values in Merged Test Data after handling:")
# print(test_data.isnull().sum())

# Encoding categorical variables
# For simplicity, let's use one-hot encoding for the 'Type' column in the store dataset
store_data_encoded = pd.get_dummies(store_data, columns=['Type'], drop_first=True)

# For simplicity, let's use one-hot encoding for the 'Date' column in the train and test dataset
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])
features_data['Date'] = pd.to_datetime(features_data['Date'])

# Extract year, month, and day from 'Date' column
train_data['Year'] = train_data['Date'].dt.year
train_data['Month'] = train_data['Date'].dt.month
train_data['Day'] = train_data['Date'].dt.day

test_data['Year'] = test_data['Date'].dt.year
test_data['Month'] = test_data['Date'].dt.month
test_data['Day'] = test_data['Date'].dt.day

features_data['Year'] = features_data['Date'].dt.year
features_data['Month'] = features_data['Date'].dt.month
features_data['Day'] = features_data['Date'].dt.day

# Drop the original 'Date' column
train_data.drop(columns=['Date'], inplace=True)
test_data.drop(columns=['Date'], inplace=True)
features_data.drop(columns=['Date'], inplace=True)

train_data = pd.merge(train_data, store_data_encoded, on='Store', how='left')
test_data = pd.merge(test_data, store_data_encoded, on='Store', how='left')

# Display the first few rows after encoding
# print("\nMerged Train Data with Encoded Store Data:")
# print(train_data.head())

# print("\nMerged Test Data with Encoded Store Data:")
# print(test_data.head())

# Assuming 'Weekly_Sales' is your target variable
X = train_data.drop(columns=['Weekly_Sales'])
y = train_data['Weekly_Sales']

# Split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize and train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train,y_train)

# Predict on the validation set
y_pred = rf_model.predict(X_test)

# Evaluate the model
wmae = mean_absolute_error(y_test, y_pred, sample_weight=(X_test['IsHoliday'].astype(int) * 5) + 1)
print("Weighted Mean Absolute Error (WMAE):", wmae)

# The WMAE value of 1652.75 gives us an idea of how well your model is performing, with lower values indicating better performance.

# Now building ARIMA model
# Assuming 'Weekly_Sales' is your target variable
y_train = train_data['Weekly_Sales']

# Fit an ARIMA model
arima_model = ARIMA(y_train, order=(1,1,1))
arima_fit = arima_model.fit()

# Predict on the validation set
arima_pred = arima_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, typ='levels')

# Evaluate the ARIMA model
wmae_arima = mean_absolute_error(y_test, arima_pred, sample_weight=(X_test['IsHoliday'].astype(int) * 5) + 1)
print("Weighted Mean Absolute Error (WMAE) ARIMA:", wmae_arima)

# The WMAE value of 15612.51 gives us an idea of how well your model is performing, with lower values indicating better performance. 
# In this case, the lower WMAE is generally better. Therefore, the Random Forest Regressor model with a lower WMAE (1652.75) 
# outperforms the ARIMA model (15612.51) on the validation set


