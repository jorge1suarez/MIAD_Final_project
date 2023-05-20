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
import optuna
from lightning.pytorch.callbacks import EarlyStopping
import json


training = None
train_dataloader = None
val_dataloader = None
def tuning():
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
    
    global train_dataloader
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )

    global val_dataloader
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
    )


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)


    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    with open("hyperparameters.json", "w") as file:
        json.dump(trial.params, file)

    print("DONE")


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 0.001, 0.1)
    hidden_size = trial.suggest_int("hidden_size", 4, 128)
    rank = trial.suggest_int("rank", 4, 30)
    rnn_layers = trial.suggest_int("rnn_layers", 1, 3)

    # Create the model
    model =  DeepAR.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        optimizer="Adam",
        loss=MultivariateNormalDistributionLoss(rank=rank),
    )
    
    
    # Create the PyTorch Lightning Trainer
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        gradient_clip_val=0.1,
        limit_train_batches=50,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10)],
    )
    

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # Return the validation loss
    print(trainer.logged_metrics["val_loss"].item())
    return trainer.logged_metrics["val_loss"].item()



    

def upload_results():
     # upload to google cloud storage bucket
    upload_blob('miad-bucket', './hyperparameters.json', f"tune/hyperparameters{datetime.datetime.now().strftime('%Y_%m_%d')}.json")


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
    tuning()
    upload_results()
