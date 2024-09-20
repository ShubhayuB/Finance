# This code tries to analyse whether BTC is a valid Maccroeconomic Indicator in the DeFI EcoSystem.
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Function to fetch historical price data using yfinance
def get_crypto_data(symbol, start_date, end_date):
    crypto = yf.Ticker(symbol)
    data = crypto.history(start=start_date, end=end_date)
    return data['Close']

# Function to calculate daily returns (price movement) from price data
def calculate_returns(price_data):
    returns = price_data.pct_change().dropna()
    return returns

# Function to calculate and plot BTC vs ETH returns with correlation coefficient
def plot_btc_vs_eth_returns(btc_returns, eth_returns):
    # Calculate Pearson correlation
    correlation, _ = pearsonr(btc_returns, eth_returns)
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot BTC and ETH returns on the same plot
    plt.plot(btc_returns.index, btc_returns, label="BTC Returns", color="blue")
    plt.plot(eth_returns.index, eth_returns, label="ETH Returns", color="orange")
    
    # Title with correlation coefficient
    plt.title(f"BTC vs ETH Returns (Correlation: {correlation:.2f})")
    
    # Add labels and legend
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Define the date range
start_date = '2022-01-01'
end_date = '2023-01-01'

# Fetch BTC and ETH price data using yfinance
btc_data = get_crypto_data('BTC-USD', start_date, end_date)
eth_data = get_crypto_data('ETH-USD', start_date, end_date)

# Calculate returns for BTC and ETH
btc_returns = calculate_returns(btc_data)
eth_returns = calculate_returns(eth_data)

# Plot BTC vs ETH returns and display the correlation coefficient
plot_btc_vs_eth_returns(btc_returns, eth_returns)
