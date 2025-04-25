import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time


def get_stock_data(ticker, start_date, end_date, max_retries=5, retry_delay=2):
    """
    Get stock data with retry mechanism to handle intermittent JSONDecodeError
    """
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                print(f"Successfully downloaded {ticker} data")
                return data
            print(f"Empty data for {ticker}, retrying {attempt+1}/{max_retries}...")
        except Exception as e:
            print(f"Error downloading {ticker} (attempt {attempt+1}/{max_retries}): {e}")
        
        
        time.sleep(retry_delay)
    
    # If all retries fail, use fallback data or raise exception
    print(f"Failed to download data for {ticker} after {max_retries} attempts")
    # Return sample data as fallback for testing
    return create_sample_data()

# Create sample data for testing if API fails
def create_sample_data():
    print("Using synthetic data for demonstration")
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)  # For reproducibility
    prices = 150 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
    sample_data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 10000000, size=len(dates))
    }, index=dates)
    return sample_data


ticker = 'AAPL'  # Example: Apple Inc.
start_date = '2023-01-01'
end_date = '2024-01-01'

try:
  
    data = get_stock_data(ticker, start_date, end_date)
    
   
    close_prices = data['Close'].dropna()
    
   
    if len(close_prices) == 0:
        raise ValueError("No valid closing prices available after processing")
        
    dates = close_prices.index
    y = close_prices.values
    X = np.arange(len(y)).reshape(-1, 1)  # Time as feature

    
    degree = 3  
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    trend = model.predict(X_poly)

    
    N = len(y)
    yf_fft = fft(y - trend)  
    freqs = fftfreq(N, d=1)  

    
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    amplitudes = np.abs(yf_fft[pos_mask])

    
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(dates, y, label='Close Price')
    plt.plot(dates, trend, label=f'Polynomial Trend (degree={degree})', linestyle='--')
    plt.title(f'{ticker} Closing Prices and Polynomial Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs, amplitudes)
    plt.title('FFT Harmonics (Spectral Amplitudes)')
    plt.xlabel('Frequency (cycles/day)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Analysis failed: {e}")
