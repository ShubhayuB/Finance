pip install yahoo_fin yfinance pandas openpyxl

import pandas as pd
from yahoo_fin import stock_info as si
import yfinance as yf
from datetime import datetime, timedelta

# Function to fetch the last 5 dividend dates
def get_last_5_dividend_dates(stock_symbol):
    dividends = si.get_dividends(stock_symbol)
    last_5_dividends = dividends.tail(5)
    return last_5_dividends.index

# Function to fetch closing prices 15 days before and after the dividend dates
def get_closing_prices(stock_symbol, dividend_dates):
    data = []
    for date in dividend_dates:
        start_date = (date - timedelta(days=15)).strftime('%Y-%m-%d')
        end_date = (date + timedelta(days=15)).strftime('%Y-%m-%d')
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        for idx, row in stock_data.iterrows():
            data.append([stock_symbol, date.strftime('%Y-%m-%d'), idx.strftime('%Y-%m-%d'), row['Close']])
    return data

def main():
    stock_symbol = 'SIEMENS.NS'  # Change this to the stock symbol you want to analyze
    dividend_dates = get_last_5_dividend_dates(stock_symbol)
    closing_prices_data = get_closing_prices(stock_symbol, dividend_dates)
    
    # Create a DataFrame
    df = pd.DataFrame(closing_prices_data, columns=['Stock Symbol', 'Dividend Date', 'Date', 'Closing Price'])
    
    # Write DataFrame to Excel
    output_file = 'dividend_stock_prices_siemens.xlsx'
    df.to_excel(output_file, index=False)
    print(f'Data has been written to {output_file}')

if __name__ == "__main__":
    main()
