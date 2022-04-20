from google.colab import files
uploaded = files.upload()
import io
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
tf.random.set_seed(51)
np.random.seed(51)
from sklearn.preprocessing import StandardScaler
from datetime import datetime

df = pd.read_csv(io.BytesIO(uploaded['Data after EV.csv']))
df = df.drop(df.columns[[0,3,4,5,6,7]], axis = 1).rename(columns = {"Total_y": "Total EV Sales"})
df.head()

tgitprint(df.shape)
df.dropna(inplace=True)
for column in df.columns[3:]:
    ssc = StandardScaler()
    df[[column]] = ssc.fit_transform(df[[column]].values)
df["Year"] = pd.to_datetime(df["Year"], format = "%Y")
df = df.set_index("Year")
grouped = df.groupby(['State'])
df.head()

df.to_numpy()[:, 2:].shape

for df in grouped:
    x_index = df[1].index.year.to_numpy()
    x_value = df[1].iloc[:,4:].to_numpy()
    y_value = df[1]["Total EV Sales"].to_numpy()

    df[1].pop("State")
    dft = df[1].T.apply(pd.to_numeric)
    test_x = dft.pop('2018-01-01')
    train_x = dft.T
  
    test_y = test_x.pop("Total EV Sales")
    train_y = train_x.pop("Total EV Sales")

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=train_x.shape[1:], activation="relu"), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
    ])

    model.summary()
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=1e-1))
    model.fit(train_x, train_y, epochs=10000, callbacks = [tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=20,
    restore_best_weights=True)])
    model.save(df[0])
    model.evaluate(tf.expand_dims(test_x, axis=0), tf.expand_dims(test_y, axis=0))

!tar -czvf models.tar.gz AR IA IL LA MN MS IN MI MO ND WI
