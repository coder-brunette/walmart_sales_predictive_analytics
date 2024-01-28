import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Merge train and features data
train_data = pd.merge(train_data, features_data, on=['Store', 'Date', 'IsHoliday'], how='left')

# Merge test and features data
test_data = pd.merge(test_data, features_data, on=['Store', 'Date', 'IsHoliday'], how='left')

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

# Visualisations

# Extracting relevant columns for analysis
# Merge train_data with features_data and store_data_encoded
sales_by_department = train_data.groupby(['Dept', 'Year', 'Month'])['Weekly_Sales'].sum().reset_index()

# Visualization 1: Total Sales by Department Over Time
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Weekly_Sales', hue='Dept', data=sales_by_department, palette='viridis')
plt.title('Total Sales by Department Over Time')
plt.xlabel('Year')
plt.ylabel('Total Weekly Sales')
plt.xticks(sales_by_department['Year'].unique(), rotation=45)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Visualization 3: Comparisons Between Departments

# Choose the top N departments to display
top_n_departments = 10  # Change this value based on your preference
avg_sales_by_dept = sales_by_department.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False)

# Take the top N departments
top_departments = avg_sales_by_dept.head(top_n_departments)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_departments.index, y=top_departments.values, palette='muted')
plt.title(f'Top {top_n_departments} Departments: Average Weekly Sales')
plt.xlabel('Department')
plt.ylabel('Average Weekly Sales')
plt.xticks(rotation=45)
plt.show()