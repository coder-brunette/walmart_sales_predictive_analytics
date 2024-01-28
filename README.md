# Walmart sales predictive analytics

# Project Title: Walmart Sales Prediction

# Project Overview:
The Walmart Sales Prediction project aimed to develop a predictive model to forecast sales for Walmart stores. Accurate sales predictions are crucial for inventory management, revenue planning, and strategic decision-making.

# Key Objectives:

1. Build a predictive model to forecast weekly sales for Walmart stores.
2. Explore and analyze seasonal trends, store variations, and the impact of holidays on sales.
3. Evaluate the performance of different models, including a Random Forest Regressor and an ARIMA time series model.

# Data Overview:

1. Utilized four main datasets: train, test, features, and store.
2. Merged and processed data, including handling missing values and encoding categorical features.
3. Extracted temporal information from the 'Date' column.

# Exploratory Data Analysis (EDA):

1. Investigated departmental and store-wise sales variations.
2. Explored the impact of store types, sizes, and holidays on sales.
3, Identified top sales periods and observed unique patterns, such as Christmas sales timing.

# Modeling:

1. # Random Forest Regressor:
Trained a Random Forest Regressor on the training set.
Evaluated performance on the validation set using Weighted Mean Absolute Error (WMAE).

2. # ARIMA Time Series Model:
Explored time series modeling using the ARIMA algorithm.
Evaluated the ARIMA model on the validation set.

# Findings:

1. The Random Forest Regressor outperformed the ARIMA model, achieving a lower WMAE on the validation set.
2. Identified departmental, regional, and seasonal patterns influencing sales.
3. Provided insights into the impact of store types, holidays, and external factors on sales variability.

# Next Steps:

1. Consider further model tuning and exploration of ensemble methods.
2. Evaluate model performance on a separate test set to assess generalization.
3. Share findings with stakeholders for potential business applications.
