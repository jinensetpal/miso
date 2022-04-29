#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

def plot_loss(history):
    """plots loss and accuracy metrics from a given training history for model evaluation

    Args:
        history: training logs
    Returns:
        None: shows plot, returns nothing
    """
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.ylim([0.994, 0.996])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

def build_model(shape):
    """returns a tensorflow model for binary classification 
    Args:
        shape: dataset input shape
    Returns:
        keras sequential model 
    """
    return tf.keras.Sequential([
    tf.keras.layers.Input(shape=(9,), name='input_features'),
    tf.keras.layers.Dense(units=20, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=40, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(units=40, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(units=20, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=10, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)])

if __name__ == '__main__':
    df = pd.read_csv('purdue-data.csv')
    df.dropna()

    X, y = df[['EXPORTER_LAG', 'SPD_LAG', 'SPD',
       'IMPORTER_LAG', 'IMPORTER', 'CRUDSCASE', 'QUICKUDS', 'BINDING',
       'CONSTRAINT_COUNT']], df['CALCTIME_CORRECT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=24)

    model = build_model(shape=(9,))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    print("Model compiled...")

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
    history = model.fit(x=X_train, y=y_train, epochs=5, validation_data=(X_valid, y_valid), callbacks=[reduce_lr], verbose=1)
    print("Model trained...")
    
    model.evaluate(X_test, y_test)
    plot_loss(history)
    model.save('outlier-classification')
