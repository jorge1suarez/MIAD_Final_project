import pandas as pd
import numpy as np
import yfinance
import yfinance as yf
import pandas_datareader as pdr
import pandas_datareader.data as web
import datetime
import lightning.pytorch as pl
import torch
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage
import joblib

import google.cloud.logging
import logging

client_logging = google.cloud.logging.Client()
client_logging.setup_logging()



def train():
    # Download Data

    date_start = '2012-01-01'
    date_end = datetime.datetime.now().strftime('%Y-%m-%d')


    tickers = ["^GSPC", 'AAPL', 'MSFT','GOOG','GOOGL','TSLA','AMZN','BRK-A','BRK-B','NVDA','META','UNH','BZ=F','NG=F', 'GC=F', 'EURUSD=X','^VIX','^IXIC']

    stock_data = yfinance.download(tickers, start=date_start, end=date_end)
    stock_data = stock_data['Close']

    for i in stock_data.columns:
        stock_data[i].fillna(method='ffill', inplace=True)
        
    stock_data = stock_data.dropna()


    fred_labels = ['EFFR', 'CSUSHPISA', 'GDP', 'CPIAUCSL', 'CPILFESL']

    fred_data = web.DataReader(fred_labels, 'fred', datetime.datetime.strptime(date_start, '%Y-%m-%d'), datetime.datetime.strptime(date_end, '%Y-%m-%d'))

    for i in fred_data.columns:
        fred_data[i].fillna(method='ffill', inplace=True)

    fred_data = fred_data.dropna()

    fred_data.index.names = ['Date']


    # Join
    data_df = stock_data.join(fred_data)

    for i in data_df.columns:
        data_df[i].fillna(method='ffill', inplace=True)
        
    data_df = data_df.resample('D').ffill()

    print(data_df.head())
    print(data_df.shape)

    # feature selection
    label = "^GSPC"
    predictors = ['AAPL', 'AMZN',  'GOOG',
        'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA', 'UNH','GC=F', 'EURUSD=X','^IXIC', 'CSUSHPISA', 'GDP', 'CPIAUCSL', 'CPILFESL']



    # Transform data

    ts_scaler = StandardScaler()

    # test - train split
    # df_train = data_df[data_df.index <= (data_df.index[-1] - datetime.timedelta(days=10))]
    df_train = data_df.copy()
    df_train = df_train.reset_index().rename(columns={'Date': 'date'})
    df_train['y'] = df_train[label]

    df_train_t = df_train[['date', 'y'] + predictors].copy()

    ts_scaler.fit(df_train[['y'] + predictors])

    df_train[['y'] + predictors] = ts_scaler.transform(df_train[['y'] + predictors])
    df_train = df_train[["date", 'y'] + predictors]


    train_series = df_train.melt(id_vars=['date'], var_name='series', value_name='value')

    train_series[['series']] = train_series[['series']].astype(str)
    train_series['value'] = pd.to_numeric(train_series['value'], downcast='float')


    min_date = train_series['date'].min()
    train_series["time_idx"] = train_series["date"].map(lambda current_date: (current_date - min_date).days)

    print(train_series.head())
    print(train_series.shape)


    # DeepVAR model

    max_prediction_length = 10
    max_encoder_length = 30


    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        train_series,
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(train_series.series)},
        group_ids=["series"],
        static_categoricals=[
            "series"
        ],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )



    batch_size = 128
    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )


    # early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
    #     callbacks=[early_stop_callback],
        limit_train_batches=50,
        enable_checkpointing=True,
    )


    net = DeepAR.from_dataset(
        training,
        learning_rate=0.006431956578942552,
        log_interval=10,
        log_val_interval=1,
        hidden_size=38,
        rnn_layers=1,
        optimizer="Adam",
        loss=MultivariateNormalDistributionLoss(rank=14),
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader
    )

    # Save artifacts

    # save standard scaler
    joblib.dump(ts_scaler, 'ts_scaler') 

    # save train_series
    train_series.to_pickle("train_series")

    # save tsdataset
    training.save("TimeSeriesDataSet_training")

    # save model in cpu
    loaded_model_in_cpu = DeepAR.load_from_checkpoint(checkpoint_path=trainer.checkpoint_callback.best_model_path, map_location=torch.device('cpu'))
    torch.save(loaded_model_in_cpu, 'model')

    print("training DONE")

def upload_model():
     # upload to google cloud storage bucket
    upload_blob('miad-bucket', './ts_scaler', f'model_{datetime.datetime.now().strftime("%Y_%m_%d")}/ts_scaler')
    upload_blob('miad-bucket', './train_series', f'model_{datetime.datetime.now().strftime("%Y_%m_%d")}/train_series')
    upload_blob('miad-bucket', './TimeSeriesDataSet_training', f'model_{datetime.datetime.now().strftime("%Y_%m_%d")}/TimeSeriesDataSet_training')
    upload_blob('miad-bucket', './model', f'model_{datetime.datetime.now().strftime("%Y_%m_%d")}/model')


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client("miadfinal")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


if __name__ == '__main__':
    train()
    upload_model()

    logging.info(f"Training done at {datetime.datetime.now().strftime('%Y_%m_%d')}, save model_{datetime.datetime.now().strftime('%Y_%m_%d')}")

