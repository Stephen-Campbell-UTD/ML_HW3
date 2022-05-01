# %%
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# %%


categoryColumns = ["Sex", "ChestPainType", "RestingECG",
                   "ExerciseAngina", "ST_Slope"]
categoryColumnsDict = {i: "category" for i in categoryColumns}
# (i)
df = pd.read_csv('./data/heart.csv').dropna()
df[categoryColumns] = df[categoryColumns].apply(
    lambda col: pd.Categorical(col).codes)


y_all = df['HeartDisease']
X_all = df.loc[:, df.columns != 'HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=1)

# %%


def p_i():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(11, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        # optimizer=tf.keras.optimizers.SGD(learning_rate=0.3),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    # %%
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        validation_data=(X_val, y_val),
    )

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, constrained_layout=True)
    # training accuracy
    ax1.plot(history.history["binary_accuracy"])
    ax1.grid(True)
    ax1.set_title("Training Accuracy")

    # testing accuracy
    ax2.set_title("Validation Accuracy")
    ax2.plot(history.history["val_binary_accuracy"])
    ax2.set_xlabel("Epoch")
    ax2.grid(True)

    plt.savefig('./images/P3_i_AccuracyvsEpoch.png')


# p_i()
# %%


y_all = df['RestingECG']
y_all = y_all.astype(np.int64)
X_all = df.loc[:, df.columns != 'RestingECG']

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.4, random_state=2)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=3)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(11, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none'),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# %%
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    batch_size=32
)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, constrained_layout=True)
# training accuracy
ax1.plot(history.history["sparse_categorical_accuracy"])
ax1.grid(True)
ax1.set_title("Training Accuracy")

# testing accuracy
ax2.set_title("Validation Accuracy")
ax2.plot(history.history["val_sparse_categorical_accuracy"])
ax2.set_xlabel("Epoch")
ax2.grid(True)

plt.savefig('./images/P3_ii_AccuracyvsEpoch.png')
# %%

tf.math.confusion_matrix(
    y_val.to_numpy(), np.argmax(model.predict(X_val), axis=1))
# %% np.argmax(model.predict(X_val), axis=1)
# type(y_val)

# %%
model.predict(X_val)
# model.predict_classes(X_val)

# %%
model.predict(X_val)
# %%
y_val

# %%
X_val.describe()
# %%
