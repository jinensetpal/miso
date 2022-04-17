#Source: 
#https://github.com/adib0073/TimeSeries-Using-TensorFlow/blob/main/Time_Series_Forecasting_with_DNN.ipynb
#https://aditya-bhattacharya.net/2020/07/11/time-series-tips-and-tricks/2/

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

plt.style.use('seaborn-darkgrid')
tf.random.set_seed(51)
np.random.seed(51)

df = pd.read_csv("data/Data after EV.csv")
df.dropna(inplace=True)
grouped = df.groupby(['State'])
for key, item in grouped:
    ## TODO: our entire program logic goes here. Each tensorflow model should be trained for the individual dataset here!
    df = grouped.get_group(key), "\n\n" # dataset to be trained

    ## TODO: Data Engineering: use scikit-learn to add StandardScaling / MinMax / One-Hot Encoding where relevant, to ensure the data is well formatted to our problem.
    ''' sample imports
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

    StandardScaler - transforms numerical elements with a standard deviation of 1.0, centered at 0. It has a range of [-1, 1]
    MinMaxScaler - transforms numerical elements onto a range of [0, 1]
    OneHotEncoder - use for categorical elements, it buckets each category into a column of it's own
    '''

    ## TODO: Set time-series steps. These are used to track the time, in our case this is on a yearly basis.

    ## TODO: Train each model. The tensorflow code you have currently implemented works well. However, ensure you set the correct target labels!

    ## TODO: Save and evaluate each model on the final set once training is complete! `model.save(key)` is a good way to do it as it ensures we know the state which is modelled.

    ''' general notes
    Model structure is good! However a few parameters to consider:
    - A higher learning rate will be helpful to get the model to converge quickly
    - tf.keras.optimizers.Adam works better than SGD in some cases, as SGD is prone to exploding gradients, especially with noisy data. Maybe try that? 
    - Good choice on the loss function!
    A good reference (https://www.tensorflow.org/tutorials/structured_data/time_series) - although this does take OOP implementation beyond the scope of what we need, it could be useful (single-step models specifically).
    '''
df.head()

# Certain Hyper-parameters to tune
split_ratio = 0.8
window_size = 60
batch_size = 64 #or 128
shuffle_buffer = 1000

def data_generate(data, window_size, batch_size, shuffle_buffer):
  '''
  Utility function for time series data generation in batches
  '''
  ts_data = tf.data.Dataset.from_tensor_slices(data)
  ts_data = ts_data.window(window_size + 1, shift=1, drop_remainder=True)
  ts_data = ts_data.flat_map(lambda window: window.batch(window_size + 1))
  ts_data = ts_data.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  ts_data = ts_data.batch(batch_size).prefetch(1)
  return ts_data

time_index = np.array(df['Year'])
data = np.array(df['Total_x'])

# Dividing into train-test split
split_index = int(split_ratio * df.shape[0])
print(split_index)

# Train-Test Split
train_data = data[:split_index]
train_time = time_index[:split_index]

test_data = data[split_index:]
test_time = time_index[split_index:]

train_dataset = data_generate(train_data, window_size, batch_size, shuffle_buffer)
test_dataset = data_generate(test_data, window_size, batch_size, shuffle_buffer)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9))
model.fit(train_dataset, epochs=200,validation_data = test_dataset)

time_int = np.array(list(range(len(data))))
time_int

forecast=[]
for time in range(len(data) - window_size):
  forecast.append(model.predict(data[time:time + window_size][np.newaxis]))

forecast = forecast[split_index-window_size:]
results = np.array(forecast)[:, 0, 0]

# Overall Error
error = tf.keras.metrics.mean_absolute_error(test_data, results).numpy()
print(error)

plt.figure(figsize=(15, 6))

plt.plot(list(range(split_index,len(data))), test_data, label = 'Test Data')
plt.plot(list(range(split_index,len(data))), results, label = 'Predictions')
#plt.fill_between(range(split_index,len(data)), results - error, results + error, alpha = 0.5, color = 'red')
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
# Plotting with Confidence Intervals
#plt.plot(list(range(split_index,len(data))), test_data, label = 'Test Data')
plt.plot(list(range(split_index,len(data))), results, label = 'Predictions', color = 'k', linestyle = '--')
plt.fill_between(range(split_index,len(data)), results - error, results + error, alpha = 0.5, color = 'red')
plt.legend()
plt.show()

# Expanding data into tensors
tensor_train_data = tf.expand_dims(train_data, axis=-1)
tensor_test_data = tf.expand_dims(test_data, axis=-1)

tensor_train_dataset = data_generate(train_data, window_size, batch_size, shuffle_buffer)
tensor_test_dataset = data_generate(test_data, window_size, batch_size, shuffle_buffer)

# Combination model of 1D CNN and LSTM
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
])

# Using callbacks to optimize the learning rates
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(tensor_train_dataset, epochs=100, callbacks=[lr_schedule])

plt.semilogx(history.history["lr"], history.history["loss"])

optimizer = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(tensor_train_dataset, epochs=200, validation_data=tensor_test_dataset)

plt.plot(list(range(200)), history.history["loss"], label = "training_loss")
plt.plot(list(range(200)), history.history["val_loss"], label = "testing_loss")
plt.legend()
plt.show()

def model_forecast(model, data, window_size):
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
  
rnn_forecast = model_forecast(model, data[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_index - window_size:-1, -1, 0]

# Overall Error
error = tf.keras.metrics.mean_absolute_error(test_data, rnn_forecast).numpy()
print(error)

plt.figure(figsize=(15, 6))

plt.plot(list(range(split_index,len(data))), test_data, label = 'Test Data')
plt.plot(list(range(split_index,len(data))), rnn_forecast, label = 'Predictions')
#plt.fill_between(range(split_index,len(data)), results - error, results + error, alpha = 0.5, color = 'red')
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
# Plotting with Confidence Intervals
#plt.plot(list(range(split_index,len(data))), test_data, label = 'Test Data')
plt.plot(list(range(split_index,len(data))), rnn_forecast, label = 'Predictions', color = 'k', linestyle = '--')
plt.fill_between(range(split_index,len(data)), rnn_forecast - error, rnn_forecast + error, alpha = 0.5, color = 'orange')
plt.legend()
plt.show()
