import yfinance
import yfinance as yf
import pandas_datareader as pdr
import pandas_datareader.data as web
import datetime
from sqlalchemy import create_engine
import re
import pandas as pd
import google.cloud.logging
import logging

client_logging = google.cloud.logging.Client()
client_logging.setup_logging()


# change url postgresql
conn_string = 'postgresql://postgres:rw,.12a@34.173.85.2/postgres'
db = create_engine(conn_string)
conn = db.connect()

date_start = '2012-01-01'
date_end = datetime.datetime.now().strftime('%Y-%m-%d')


print(f"Downloading data from:{date_start} to:{date_end}")

tickers = ["^GSPC", 'AAPL', 'MSFT','GOOG','GOOGL','TSLA','AMZN','BRK-A','BRK-B','NVDA','META','UNH','BZ=F','NG=F', 'GC=F', 'EURUSD=X','^VIX','^IXIC']

from_date = None
to_date = None

for ticker in tickers:
    print(f"Downloading data for ticker: {ticker}")
    stock_data = yfinance.download(ticker, start=date_start, end=date_end)
    stock_data = stock_data.dropna()
    df = stock_data.reset_index().rename(columns={'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close':'adj_close', 'Volume':'volume'})
    from_date = df['date'].min()
    to_date = df['date'].max()

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
date_range = pd.date_range(start=df['date'].max()+pd.Timedelta(days=1), end=df['date'].max()+pd.Timedelta(days=10), freq='D')
df = pd.concat([df, pd.DataFrame({'date': date_range})], ignore_index=True)

df.to_sql('date_table', con=conn, if_exists='replace', index=False)
conn.commit()
print(f"Wrote data to Table date_table, rows: {df.shape[0]}")


print("Done")
logging.info(f"ETL done at {datetime.datetime.now().strftime('%Y_%m_%d')}, \
             days downloaded (from-to): {from_date} - {to_date}")

conn.close()