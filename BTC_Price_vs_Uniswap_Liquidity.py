import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
file_path = 'C:/Users/shubh/Documents/Bits Pilani/Sperax/USDC-ETH Pool Volume(1 Month).xlsx' 
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Calculate daily percentage change (volatility) for BTC price and USDC/ETH pool volume
df['BTC Price Change (%)'] = df['BTC Price (In Dollars)'].pct_change() * 100
df['Pool Volume Change (%)'] = df['Pool Volume ( In Dollars)'].pct_change() * 100

# Drop the first row with NaN values
df_volatility = df.dropna(subset=['BTC Price Change (%)', 'Pool Volume Change (%)'])

# Calculate the correlation coefficient
correlation_coefficient = df_volatility[['BTC Price Change (%)', 'Pool Volume Change (%)']].corr().iloc[0, 1]
print(f'Correlation Coefficient: {correlation_coefficient}')

# Plot the volatilities for comparison
plt.figure(figsize=(12, 6))

# Plot BTC Price Volatility
plt.plot(df_volatility['Date'], df_volatility['BTC Price Change (%)'], label='BTC Price Volatility (%)', color='blue')

# Plot USDC/ETH Pool Volume Volatility
plt.plot(df_volatility['Date'], df_volatility['Pool Volume Change (%)'], label='USDC/ETH Pool Volume Volatility (%)', color='green')


# Adding labels and title
plt.title(f"BTC Price Volatility vs USDC/ETH Pool Volume Volatility (Correlation: {correlation_coefficient:.2f})")
plt.xlabel('Date')
plt.ylabel('Percentage Change (%)')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
