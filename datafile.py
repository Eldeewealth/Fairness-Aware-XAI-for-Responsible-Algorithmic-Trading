import yfinance as yf

# Define the ticker symbol for the S&P 500 index
ticker = "^GSPC"

# Download historical data
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Save to CSV
data.to_csv("sp500_data.csv")

print("✅ S&P 500 data saved as 'sp500_data.csv'")

