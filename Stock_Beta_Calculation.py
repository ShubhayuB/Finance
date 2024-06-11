import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Initilization 
siemens_ticker = 'SIEMENS.NS'  # I have taken Siemens here. You can use choose ticker of your choice to analyse
nifty_ticker = '^NSEI'    # NIFTY 50 Index
start_date = '2020-01-01'
end_date = '2024-01-01' #5 year analysis
siemens_data = yf.download(siemens_ticker, start=start_date, end=end_date)
nifty_data = yf.download(nifty_ticker, start=start_date, end=end_date)

# Calculating daily returns
siemens_data['Returns'] = siemens_data['Adj Close'].pct_change()
nifty_data['Returns'] = nifty_data['Adj Close'].pct_change()

# Data cleaning and aligning
siemens_returns = siemens_data['Returns'].dropna()
nifty_returns = nifty_data['Returns'].dropna()
aligned_data = pd.concat([siemens_returns, nifty_returns], axis=1).dropna()
aligned_data.columns = ['Siemens Returns', 'NIFTY Returns']

#linear regression estimation
X = aligned_data['NIFTY Returns']
y = aligned_data['Siemens Returns']
X = sm.add_constant(X)  # Add a constant term to the predictor
model = sm.OLS(y, X).fit()
beta = model.params['NIFTY Returns']


print(f'The beta of Siemens relative to the NIFTY index is: {beta:.4f}')

# Plotting the regression
plt.figure(figsize=(10, 6))
plt.scatter(aligned_data['NIFTY Returns'], aligned_data['Siemens Returns'], label='Data points')
plt.plot(aligned_data['NIFTY Returns'], model.fittedvalues, color='red', label='Fitted line')
plt.xlabel('NIFTY Returns')
plt.ylabel('Siemens Returns')
plt.title('Siemens vs NIFTY Returns')
plt.legend()
plt.show()
