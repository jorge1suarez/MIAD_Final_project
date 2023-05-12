import yfinance
import yfinance as yf
import pandas_datareader as pdr
import pandas_datareader.data as web
import datetime
from sqlalchemy import create_engine
import re
import psycopg2

# change url postgresql
conn_string = 'postgresql://postgres:rw,.12a@34.173.85.2/postgres'
db = create_engine(conn_string)
conn = db.connect()
conn_psycopg2 = psycopg2.connect(database="postgres",
                        host="192.168.217.110",
                        user="postgres",
                        password="rw,.12a",
                        port="5432")

cur = conn_psycopg2.cursor()


now = datetime.datetime.now()
date_end = now.strftime('%Y-%m-%d')
date_start = (now - datetime.timedelta(days=2)).strftime('%Y-%m-%d')

print(f"Downloading data from:{date_start} to:{date_end}") # 2 days before.

tickers = ["^GSPC", 'AAPL', 'MSFT','GOOG','GOOGL','TSLA','AMZN','BRK-A','BRK-B','NVDA','META','UNH','BZ=F','NG=F', 'GC=F', 'EURUSD=X','^VIX','^IXIC']

for ticker in tickers:
    print(f"Downloading data for ticker: {ticker}")
    stock_data = yfinance.download(ticker, start=date_start, end=date_end)
    stock_data = stock_data.dropna()
    df = stock_data.reset_index().rename(columns={'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close':'adj_close', 'Volume':'volume'})
    

    table_name = "ts_" + re.sub(r'\W+', '', ticker).lower() # remove special characters from ticker and lower case

    cur.execute(f"SELECT * FROM {table_name} ORDER BY date DESC LIMIT 1")
    last_record = cur.fetchone()
    df = df[df['date'] > last_record[0]]

    df.to_sql(table_name, con=conn, if_exists='append', index=False)
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

    cur.execute(f"SELECT * FROM {table_name} ORDER BY date DESC LIMIT 1")
    last_record = cur.fetchone()
    df = df[df['date'] > last_record[0]]

    df.to_sql(table_name, con=conn, if_exists='append', index=False)
    conn.commit()
    print(f"Wrote data to Table {table_name}, rows: {df.shape[0]}")

print("Done")
cur.close()
conn_psycopg2.close()
conn.close()