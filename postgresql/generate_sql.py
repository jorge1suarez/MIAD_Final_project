
import re
tickers = ["^GSPC", 'AAPL', 'MSFT','GOOG','GOOGL','TSLA','AMZN','BRK-A','BRK-B','NVDA','META','UNH','BZ=F','NG=F', 'GC=F', 'EURUSD=X','^VIX','^IXIC']


fred_labels = ['EFFR', 'CSUSHPISA', 'GDP', 'CPIAUCSL', 'CPILFESL']

sql = ""

# for historical data
for ticker in tickers:
    table_name = "ts_" + re.sub(r'\W+', '', ticker).lower() # remove special characters from ticker and lower case

    sql += f'DROP TABLE "{table_name}";\n'
    sql += f'CREATE TABLE "{table_name}" ( \n' + \
    '\t"date" date primary key, \n' + \
    '\t"open" real, \n' + \
    '\t"high" real, \n' + \
    '\t"low" real, \n' + \
    '\t"close" real, \n' + \
    '\t"adj_close" real, \n' + \
    '\t"volume" real \n' + \
    '); \n\n'


for label in fred_labels:
    table_name = "ts_" + re.sub(r'\W+', '', label).lower() # remove special characters from ticker and lower case

    sql += f'DROP TABLE "{table_name}";\n'
    sql += f'CREATE TABLE "{table_name}" ( \n' + \
    '\t"date" date primary key, \n' + \
    '\t"value" real \n' + \
    '); \n\n'


# for predictions


for ticker in tickers:
    table_name = "ts_" + re.sub(r'\W+', '', ticker).lower() + "_pred" # remove special characters from ticker and lower case

    sql += f'DROP TABLE "{table_name}";\n'
    sql += f'CREATE TABLE "{table_name}" ( \n' + \
    '\t"date" date primary key, \n' + \
    '\t"mean" real, \n' + \
    '\t"min" real, \n' + \
    '\t"max" real, \n' + \
    '\t"q2_5" real, \n' + \
    '\t"q5" real, \n' + \
    '\t"q25" real, \n' + \
    '\t"q50" real, \n' + \
    '\t"q75" real, \n' + \
    '\t"q95" real, \n' + \
    '\t"q97_5" real \n' + \
    '); \n\n'


for label in fred_labels:
    table_name = "ts_" + re.sub(r'\W+', '', label).lower() + "_pred"# remove special characters from ticker and lower case

    sql += f'DROP TABLE "{table_name}";\n'
    sql += f'CREATE TABLE "{table_name}" ( \n' + \
    '\t"date" date primary key, \n' + \
    '\t"mean" real, \n' + \
    '\t"min" real, \n' + \
    '\t"max" real, \n' + \
    '\t"q2_5" real, \n' + \
    '\t"q5" real, \n' + \
    '\t"q25" real, \n' + \
    '\t"q50" real, \n' + \
    '\t"q75" real, \n' + \
    '\t"q95" real, \n' + \
    '\t"q97_5" real \n' + \
    '); \n\n'



with open('init.sql', 'w') as file:
    file.write(sql)
