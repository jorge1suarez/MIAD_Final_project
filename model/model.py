import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import torch

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss


max_prediction_length = 10
max_encoder_length = 30
batch_size = 128

#training_cutoff = train_series["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

import joblib

# Load scaler
ts_scaler = joblib.load('ts_scaler.save') 
print(ts_scaler)

# load pandas dataframe training
train_series = pd.read_pickle('train_series')
print(train_series.head())

# torch ts dataset

training = TimeSeriesDataSet.load("TimeSeriesDataSet_training")
print(training)


training_cutoff = train_series["time_idx"].max() - max_prediction_length

validation = TimeSeriesDataSet.from_dataset(training, train_series, min_prediction_idx=training_cutoff + 1)

train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)


# load train model
best_model = torch.load('model_v2')
print(best_model)



## predictions
prediction = best_model.predict(
    val_dataloader, mode="raw", return_index=True, trainer_kwargs=dict(accelerator="cpu")
)

print(prediction)