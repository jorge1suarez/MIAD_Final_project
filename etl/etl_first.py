import yfinance
import yfinance as yf
import pandas_datareader as pdr
import pandas_datareader.data as web
import datetime
from sqlalchemy import create_engine
import re


# change url postgresql
conn_string = 'postgresql://postgres:rw,.12a@34.173.85.2/postgres'
db = create_engine(conn_string)
conn = db.connect()

date_start = '2012-01-01'
date_end = datetime.datetime.now().strftime('%Y-%m-%d')


print(f"Downloading data from:{date_start} to:{date_end}")

tickers = ["^GSPC", 'AAPL', 'MSFT','GOOG','GOOGL','TSLA','AMZN','BRK-A','BRK-B','NVDA','META','UNH','BZ=F','NG=F', 'GC=F', 'EURUSD=X','^VIX','^IXIC']

for ticker in tickers:
    print(f"Downloading data for ticker: {ticker}")
    stock_data = yfinance.download(ticker, start=date_start, end=date_end)
    stock_data = stock_data.dropna()
    df = stock_data.reset_index().rename(columns={'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close':'adj_close', 'Volume':'volume'})
    

    table_name = "ts_" + re.sub(r'\W+', '', ticker).lower() # remove special characters from ticker and lower case

    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
    conn.commit()
    print(f"Wrote data to Table {table_name}, rows: {df.shape[0]}")


fred_labels = ['EFFR', 'CSUSHPISA', 'GDP', 'CPIAUCSL', 'CPILFESL']

for label in fred_labels:
    print(f"Downloading data for ticker: {label}")
    fred_data = web.DataReader(label, 'fred', datetime.datetime.strptime(date_start, '%Y-%m-%d'), datetime.datetime.strptime(date_end, '%Y-%m-%d'))
    fred_data[label].fillna(method='ffill', inplace=True)
    fred_data = fred_data.dropna()
    fred_data.index.names = ['Date']
    df = fred_data.reset_index().rename(columns={'Date':'date', label:'value'})

    table_name = "ts_" + re.sub(r'\W+', '', label).lower() # remove special characters from ticker and lower case

    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
    conn.commit()
    print(f"Wrote data to Table {table_name}, rows: {df.shape[0]}")


# date table
df = stock_data.reset_index().rename(columns={'Date':'date'})[['date']]
df.to_sql('date_table', con=conn, if_exists='replace', index=False)
conn.commit()
print(f"Wrote data to Table date_table, rows: {df.shape[0]}")


print("Done")
conn.close()