import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)
def simple_moving_window(signal,window_size):
    import numpy as np
    windowed_signal = []
    for i in range(len(signal)-(window_size-1)):
        window = signal[i:i+window_size]
        windowed_signal.append(window)
    return np.array(windowed_signal)
def split_and_shuffle(windowed_signal):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(windowed_signal, test_size=0.2,shuffle=False)
    return train_df,test_df 
def feature_label_split(windowed_signal,prediction_length):
    from numpy import array
    features = []
    labels = []
    for window in windowed_signal:
        features.append(window[:-prediction_length])
        labels.append(window[-prediction_length:])
    return array(features),array(labels)
x = np.linspace(0,999,1000)
y = np.sin(x/50)
"""
window size has to be bigger than prediction length. just because of my engineering. you could change variables.
"""
prediction_length = 80
window_size = 100

y_windowed = simple_moving_window(signal=y,window_size=window_size)
train,test = split_and_shuffle(windowed_signal=y_windowed)
test_features,test_labels = feature_label_split(test,prediction_length=prediction_length)
train_features,train_labels = feature_label_split(train,prediction_length=prediction_length)
number_of_hidden_layer_neurons = 100
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=number_of_hidden_layer_neurons, activation='relu',input_shape=(train_features.shape[-1],)),
    tf.keras.layers.Dense(prediction_length)
])

MAX_EPOCHS = 100

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=2,
                                                mode='min')

model.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()])

history = model.fit(train_features,train_labels, epochs=MAX_EPOCHS,
                    callbacks=[early_stopping])
fig,axes = plt.subplots(1,1,figsize=(5,5),dpi=200)
plt.plot(x,y)

predictions = model.predict(test_features)

train_size = train.shape[0]
feature_id = 100
plt.plot(x[train_size+feature_id:train_size+feature_id+(window_size-prediction_length)],test_features[feature_id])

plt.plot(x[train_size+feature_id+(window_size-prediction_length):train_size+feature_id+window_size],predictions[feature_id])
