import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Step 1: Download Historical Stock Data
# Define stock ticker, market index ticker, and download data
stock_ticker = 'AAPL' 
market_ticker = '^GSPC' 
risk_free_ticker = '^IRX'  

# Download historical data from Yahoo Finance
stock_data = yf.download(stock_ticker, start='2020-01-01', end='2023-01-01')['Adj Close']
market_data = yf.download(market_ticker, start='2020-01-01', end='2023-01-01')['Adj Close']
rf_data = yf.download(risk_free_ticker, start='2020-01-01', end='2023-01-01')['Adj Close']

# Step 2: Calculate Daily Returns
stock_returns = stock_data.pct_change().dropna()
market_returns = market_data.pct_change().dropna()
rf_rate = rf_data.pct_change().dropna() / 100  

# Align data by common dates
returns_data = pd.concat([stock_returns, market_returns, rf_rate], axis=1).dropna()
returns_data.columns = ['Stock', 'Market', 'RiskFree']

# Calculate excess returns
returns_data['Excess_Stock'] = returns_data['Stock'] - returns_data['RiskFree']
returns_data['Excess_Market'] = returns_data['Market'] - returns_data['RiskFree']

# Step 3: Apply Normalized Weights
lambda_value = 0.  # Example lambda value
t_values = np.arange(len(returns_data))  # Time values

# Calculate weights and normalized weights
weights = lambda_value ** t_values
normalized_weights = weights / weights.sum()

# Step 4: Perform Weighted Regression
X = returns_data['Excess_Market']
y = returns_data['Excess_Stock']

# Add constant for the intercept
X = sm.add_constant(X)

# Fit the weighted least squares model
wls_model = sm.WLS(y, X, weights=normalized_weights).fit()

# Extract Beta (the coefficient of the Excess_Market term)
beta = wls_model.params['Excess_Market']
print(f"Beta value: {beta}")

# Display the regression results
print(wls_model.summary())
