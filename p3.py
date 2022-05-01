# %%
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('./data/housing.csv').dropna()
y_all = df['median_house_value'].to_numpy(dtype=np.float32)
X_all = df.loc[:, df.columns !=
               'median_house_value'].to_numpy(dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=1)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(11, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)

# %%
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_val, y_val),
)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, constrained_layout=True)
# training accuracy
ax1.plot(history.history["root_mean_squared_error"])
ax1.grid(True)
ax1.set_title("Training Accuracy")

# testing accuracy
ax2.set_title("Validation Root Mean Squared Error")
ax2.plot(history.history["val_root_mean_squared_error"])
ax2.set_xlabel("Epoch")
ax2.grid(True)

plt.savefig('./images/P3_i_AccuracyvsEpoch.png')

# %%
print(history.history)

# %%
