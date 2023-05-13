# import lightning.pytorch as pl
import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
# from pytorch_forecasting.data import NaNLabelEncoder
# from pytorch_forecasting.data.examples import generate_ar_data
# from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss
import joblib
from sqlalchemy import create_engine
import re


max_prediction_length = 10
max_encoder_length = 30
batch_size = 128

#training_cutoff = train_series["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length



# Load scaler
ts_scaler = joblib.load('ts_scaler_v3') 

# load pandas dataframe training
train_series = pd.read_pickle('train_series_v3')
print(train_series.tail())
# torch ts dataset
training = TimeSeriesDataSet.load("TimeSeriesDataSet_training_v3")

# load train model
best_model = torch.load('model_v3')
print(best_model)



# predict future 10 days

date_range = pd.date_range(start=train_series['date'].max()+pd.Timedelta(days=1), end=train_series['date'].max()+pd.Timedelta(days=10), freq='D')
idx_range = np.arange(train_series['time_idx'].max()+1, train_series['time_idx'].max()+1 + 10)

train_series_new = train_series.copy()

for i in train_series['series'].unique():

    new_data = {
        'date': date_range,
        'series': [i]*max_prediction_length,
        'value': [0.0]*max_prediction_length,
        'time_idx': idx_range
    }
    
    new_df = pd.DataFrame(new_data)

    train_series_new = pd.concat([train_series_new, new_df], ignore_index=True)


training_cutoff = train_series_new["time_idx"].max() - max_prediction_length


# dataloader

validation = TimeSeriesDataSet.from_dataset(training, train_series_new, min_prediction_idx=training_cutoff + 1)

val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)



## predict 
prediction = best_model.predict(
    val_dataloader, mode="raw", return_index=True, trainer_kwargs=dict(accelerator="cpu")
)

# ts_columns = list(train_series['series'].unique()) # orden importa para el scaler
ts_columns = list(ts_scaler.feature_names_in_)


df_test_mean = pd.DataFrame()
df_test_lower = pd.DataFrame()
df_test_upper = pd.DataFrame()

df_test_q10 = pd.DataFrame()
df_test_q90 = pd.DataFrame()
df_test_q25 = pd.DataFrame()
df_test_q75 = pd.DataFrame()
df_test_q50 = pd.DataFrame()


for index, row in prediction.index.iterrows():
    ts_predict = np.array(prediction.output.prediction[index, :])
    df_test_mean[str(row['series'])] = ts_predict.mean(axis=1)
    df_test_lower[str(row['series'])] = ts_predict.min(axis=1)
    df_test_upper[str(row['series'])] = ts_predict.max(axis=1)
    
    #90% prediction interval
    df_test_q10[str(row['series'])] = np.quantile(ts_predict, q=0.1, axis = 1)
    df_test_q90[str(row['series'])] = np.quantile(ts_predict, q=0.9, axis = 1)

    #50% prediction interval
    df_test_q25[str(row['series'])] = np.quantile(ts_predict, q=0.25, axis = 1)
    df_test_q75[str(row['series'])] = np.quantile(ts_predict, q=0.75, axis = 1)
    
    # median
    df_test_q50[str(row['series'])] = np.quantile(ts_predict, q=0.5, axis = 1)


    
df_test_mean[ts_columns] = ts_scaler.inverse_transform(df_test_mean[ts_columns].values)
df_test_lower[ts_columns] = ts_scaler.inverse_transform(df_test_lower[ts_columns].values)
df_test_upper[ts_columns] = ts_scaler.inverse_transform(df_test_upper[ts_columns].values)

df_test_q10[ts_columns] = ts_scaler.inverse_transform(df_test_q10[ts_columns].values)
df_test_q90[ts_columns] = ts_scaler.inverse_transform(df_test_q90[ts_columns].values)
df_test_q25[ts_columns] = ts_scaler.inverse_transform(df_test_q25[ts_columns].values)
df_test_q75[ts_columns] = ts_scaler.inverse_transform(df_test_q75[ts_columns].values)
df_test_q50[ts_columns] = ts_scaler.inverse_transform(df_test_q50[ts_columns].values)

df_test_mean['date'] = date_range.values
df_test_lower['date'] = date_range.values
df_test_upper['date'] = date_range.values

df_test_q10['date'] = date_range.values
df_test_q90['date'] = date_range.values
df_test_q25['date'] = date_range.values
df_test_q75['date'] = date_range.values
df_test_q50['date'] = date_range.values

print(df_test_q50.tail())

# save predictions


conn_string = 'postgresql://postgres:rw,.12a@34.173.85.2/postgres'
db = create_engine(conn_string)
conn = db.connect()

print("saving predictions ...")


for ts in ts_columns:
    if ts == 'y': # SP500
        table_name = "ts_" + re.sub(r'\W+', '', '^GSPC').lower() + "_pred"# remove special characters from ticker and lower case
        
        df = pd.DataFrame()
        df['date'] = df_test_mean['date']
        df['mean'] = df_test_mean[ts]
        df['min'] = df_test_lower[ts]
        df['max'] = df_test_upper[ts]
        df['q10'] = df_test_q10[ts]
        df['q25'] = df_test_q25[ts]
        df['q50'] = df_test_q50[ts]
        df['q75'] = df_test_q75[ts]
        df['q90'] = df_test_q90[ts]

        df.to_sql(table_name, con=conn, if_exists='replace', index=False)
        conn.commit()
        print(f"Wrote data to Table {table_name}")

    else:
        table_name = "ts_" + re.sub(r'\W+', '', ts).lower() + "_pred"# remove special characters from ticker and lower case
        
        df = pd.DataFrame()
        df['date'] = df_test_mean['date']
        df['mean'] = df_test_mean[ts]
        df['min'] = df_test_lower[ts]
        df['max'] = df_test_upper[ts]
        df['q10'] = df_test_q10[ts]
        df['q25'] = df_test_q25[ts]
        df['q50'] = df_test_q50[ts]
        df['q75'] = df_test_q75[ts]
        df['q90'] = df_test_q90[ts]

        df.to_sql(table_name, con=conn, if_exists='replace', index=False)
        conn.commit()
        print(f"Wrote data to Table {table_name}")
        


print("Done")
conn.close()