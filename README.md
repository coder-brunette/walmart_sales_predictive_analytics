# walmart_sales_predictive_analytics

Project Title: Walmart Sales Prediction

Project Overview:
The Walmart Sales Prediction project aimed to develop a predictive model to forecast sales for Walmart stores. Accurate sales predictions are crucial for inventory management, revenue planning, and strategic decision-making.

Key Objectives:

Build a predictive model to forecast weekly sales for Walmart stores.
Explore and analyze seasonal trends, store variations, and the impact of holidays on sales.
Evaluate the performance of different models, including a Random Forest Regressor and an ARIMA time series model.
Data Overview:

Utilized four main datasets: train, test, features, and store.
Merged and processed data, including handling missing values and encoding categorical features.
Extracted temporal information from the 'Date' column.
Exploratory Data Analysis (EDA):

Investigated departmental and store-wise sales variations.
Explored the impact of store types, sizes, and holidays on sales.
Identified top sales periods and observed unique patterns, such as Christmas sales timing.
Modeling:

Random Forest Regressor:
Trained a Random Forest Regressor on the training set.
Evaluated performance on the validation set using Weighted Mean Absolute Error (WMAE).
ARIMA Time Series Model:
Explored time series modeling using the ARIMA algorithm.
Evaluated the ARIMA model on the validation set.
Findings:

The Random Forest Regressor outperformed the ARIMA model, achieving a lower WMAE on the validation set.
Identified departmental, regional, and seasonal patterns influencing sales.
Provided insights into the impact of store types, holidays, and external factors on sales variability.
Next Steps:

Consider further model tuning and exploration of ensemble methods.
Evaluate model performance on a separate test set to assess generalization.
Share findings with stakeholders for potential business applications.
