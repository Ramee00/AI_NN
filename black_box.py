import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


ds = pd.read_csv(r"train.csv")

ds = np.array(ds)
m, n = ds.shape
np.random.shuffle(ds)

test = ds[0:1000].T
Y_test = test[0]
X_test = test[1:n]
X_test = X_test / 255.

data_train = ds[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

model = Sequential()
model.add(Dense(20, input_shape=(10468,), activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import tensorflow as tf

X_train = tf.stack(X_train)
Y_train = tf.stack(Y_train)
model.fit(X_train, Y_train, epochs= 3000, batch_size=10)
X_train.shape
X_test.shape
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))
model.summary()
model.predict(X_test , verbose = 1)
