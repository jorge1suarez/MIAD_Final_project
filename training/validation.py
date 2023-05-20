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
import json
import time

import google.cloud.logging
import logging
from google.cloud import monitoring_v3
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re

client_logging = google.cloud.logging.Client()
client_logging.setup_logging()



def validation():
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

    training_cutoff = train_series["time_idx"].max() - max_prediction_length

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    global training
    training = TimeSeriesDataSet(
        train_series[lambda x: x.time_idx <= training_cutoff],
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


    # for validation
    validation = TimeSeriesDataSet.from_dataset(training, train_series, min_prediction_idx=training_cutoff + 1)

    batch_size = 128
    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )


    # early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
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
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # metrics

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = DeepAR.load_from_checkpoint(best_model_path)

    df_test = df_train_t[df_train_t['date'] > (df_train_t['date'].iloc[-1] - datetime.timedelta(days=10))]

    prediction = best_model.predict(
        val_dataloader, mode="raw", return_index=True, trainer_kwargs=dict(accelerator="cpu")
    )

    df_test_pred = pd.DataFrame()
    

    for index, row in prediction.index.iterrows():
        ts_predict = np.array(prediction.output.prediction[index, :])
        # df_test_pred[str(row['series'])] = ts_predict.mean(axis=1)
        df_test_pred[str(row['series'])] = np.quantile(ts_predict, q=0.5, axis = 1)
        
    df_test_pred[df_test.drop(columns=['date']).columns] = ts_scaler.inverse_transform(df_test_pred[df_test.drop(columns=['date']).columns].values)
    df_test_pred['date'] = df_test['date'].values


    metrics = {}
    for col in df_test.drop(columns=['date']).columns:
        print(col, ":")
        
        mae = mean_absolute_error(df_test[col], df_test_pred[col])
        mse = mean_squared_error(df_test[col], df_test_pred[col])
        rmse = mean_squared_error(df_test[col], df_test_pred[col], squared=False)

        mape = mape_metric(df_test[col].values, df_test_pred[col].values)
        smape = smape_metric(df_test[col].values, df_test_pred[col].values)

        print("MAPE:", mape)
        print("SMAPE:", smape)
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print("\n")
        metrics[col] = {"mape": mape, "smape": smape, "mae": mae, "mse": mse, "rmse": rmse}


    # save to file
    with open("metrics.json", "w") as file:
        json.dump(metrics, file)

    upload_results()

    for key, value in metrics.items():
        for metric, metric_value in value.items():
            if key == 'y': # SP500
                logging.info(f"Validation {'SP500'} {metric}: {metric_value}")
            else:
                logging.info(f"Validation {key} {metric}: {metric_value}")

    # send metrics
    # for key, value in metrics.items():

    #     for metric, metric_value in value.items():

    #         if key == 'y': # SP500
    #             metric_name = re.sub(r'\W+', '', '^GSPC').lower() + "_" + metric
    #         else:
    #             metric_name = re.sub(r'\W+', '', key).lower() + "_" + metric
            
    #         metric_type = f"custom.googleapis.com/{metric_name}"

    #         send_custom_metric(metric_type, metric_value)

    
    print("validation DONE")


def upload_results():
     # upload to google cloud storage bucket
    upload_blob('miad-bucket', './metrics.json', f"validation/metrics_{datetime.datetime.now().strftime('%Y_%m_%d')}.json")


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


def send_custom_metric(metric_type, metric_value):
    """Sends a custom metric to Google Cloud Monitoring."""
    
    client_metrics = monitoring_v3.MetricServiceClient()
    project_id = "miadfinal"
    project_name = f"projects/{project_id}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = metric_type
    series.metric.labels["serie"] = "miad"
    series.resource.type = "gce_instance"
    series.resource.labels["instance_id"] = "410681636358977934"
    series.resource.labels["zone"] = "us-central1-a"

    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10 ** 9)
    interval = monitoring_v3.TimeInterval(
        {"end_time": {"seconds": seconds, "nanos": nanos}}
    )

    point = monitoring_v3.Point({"interval": interval, "value": {"double_value": metric_value}})
    series.points = [point]

    client_metrics.create_time_series(request={"name": project_name, "time_series": [series]})
    print(f"Successfully wrote time series {metric_type}.")
    time.sleep(5)



def mape_metric(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape_metric(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100



if __name__ == '__main__':
    validation()
    #upload_results()
    logging.info(f"Validation done at {datetime.datetime.now().strftime('%Y_%m_%d')}")

