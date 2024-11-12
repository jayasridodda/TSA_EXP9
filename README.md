
## Developed By: Dodda Jayasri
## Reg No: 212222240028
## Date: 

# EX.NO.09        A project on Time series analysis on ratings rate forecasting using ARIMA model in python


### AIM:
The aim of this project is to forecast ratings rate in Salesforcehistory dataset using the ARIMA model and evaluate its accuracy through visualization and statistical metrics.

### ALGORITHM:
Here's a condensed 5-point version for applying ARIMA on the Goodreads dataset:

1. Load and Prepare Data: Load `Salesforcehistory.csv`, convert `Date` to datetime, and set it as the index.

2. Initial Visualization and Stationarity Check: Plot the time series; use ADF, ACF, and PACF to assess stationarity.

3. Apply Differencing: If non-stationary, apply differencing to stabilize the series.

4. Select ARIMA Parameters: Use `auto_arima` or ACF/PACF plots to find suitable `(p, d, q)` values.

5. Fit Model, Forecast, and Evaluate: Fit the ARIMA model, make predictions, and evaluate using MAE, RMSE, and comparison plots.

   
### PROGRAM:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('Salesforcehistory.csv')
# Ensure the date column is in the correct format; adjust the column name as needed
data['Date'] = pd.to_datetime(data['Date'])  
data.set_index('Date', inplace=True)

# Visualize the time series data to inspect trends
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Sale Price')  # Adjust column name as needed, e.g., 'Close'
plt.title('Time Series of Sale Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Check stationarity using the ADF test
result = adfuller(data['Close'])  # Adjust column name as needed
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, apply differencing to achieve stationarity
data['close_diff'] = data['Close'].diff().dropna()  # Adjust column name as needed
result_diff = adfuller(data['close_diff'].dropna())
print('Differenced ADF Statistic:', result_diff[0])
print('Differenced p-value:', result_diff[1])

# Plot ACF and PACF for the differenced data
plot_acf(data['close_diff'].dropna())
plt.title('ACF of Differenced Sale Price')
plt.show()

plot_pacf(data['close_diff'].dropna())
plt.title('PACF of Differenced Sale Price')
plt.show()

# Plot Differenced Representation
plt.figure(figsize=(10, 5))
plt.plot(data['close_diff'], label='Differenced Stock Price', color='red')
plt.title('Differenced Representation of Stock Price')
plt.xlabel('Date')
plt.ylabel('Differenced Stock Price')
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.legend()
plt.show()

# Use auto_arima to find the optimal (p, d, q) parameters
stepwise_model = auto_arima(data['Close'], start_p=1, start_q=1,
                            max_p=3, max_q=3, seasonal=False, trace=True)
p, d, q = stepwise_model.order
print(stepwise_model.summary())

# Fit the ARIMA model using the optimal parameters
model = sm.tsa.ARIMA(data['Close'], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 30 days (or adjust period based on your needs)
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual Sale Price')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('ARIMA Forecast of Stock Price')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['Close']) - 1)
mae = mean_absolute_error(data['Close'], predictions)
rmse = np.sqrt(mean_squared_error(data['Close'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/c6991ea3-8a79-48fc-8920-f1a81ba428fc)

![image](https://github.com/user-attachments/assets/4ae60cff-3029-4ab3-8da3-7b0575715a02)

![image](https://github.com/user-attachments/assets/37f95db1-c3de-4807-9d40-dedc9ea571c2)

![image](https://github.com/user-attachments/assets/1b31b331-ced9-4772-b668-33d9fd1ea860)

![image](https://github.com/user-attachments/assets/606c220e-7d62-40a1-a549-ebe3e31b1d43)

![image](https://github.com/user-attachments/assets/beeff65b-55b0-4c9b-bfc5-31e8e88355d9)


### RESULT:
Thus the Time series analysis on Ratings rate in Goodreads Books prediction using the ARIMA model completed successfully.
