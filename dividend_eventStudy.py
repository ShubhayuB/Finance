pip install yfinance pandas openpyxl requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf

# Define the URL
url = "https://www.moneycontrol.com/company-facts/siemens/dividends/S"

# Fetch the web page
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# Extract the last 5 dividend announcement dates
dividend_table = soup.find("table", {"class": "mctable1"})
rows = dividend_table.find_all("tr")[1:6]  # Get the last 5 rows
dividend_dates = [row.find_all("td")[0].text.strip() for row in rows]

# Convert dividend_dates to datetime format
dividend_dates = pd.to_datetime(dividend_dates, format="%d-%m-%Y")

print(dividend_dates)

# Define the stock ticker
ticker = "SIEMENS.NS"

# Initialize an empty DataFrame to store the data
all_data = pd.DataFrame()

for date in dividend_dates:
    start_date = (date - pd.DateOffset(days=15)).strftime("%Y-%m-%d")
    end_date = (date + pd.DateOffset(days=15)).strftime("%Y-%m-%d")
    
    # Fetch the historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Add the announcement date to the DataFrame
    data['Announcement Date'] = date
    
    # Append to the main DataFrame
    all_data = all_data.append(data)

# Reset index
all_data.reset_index(inplace=True)

print(all_data)

# Save the DataFrame to an Excel file
all_data.to_excel("Siemens_Dividend_Data.xlsx", index=False)
