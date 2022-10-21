import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


data = pd.read_csv(r"/content/train.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
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
model.save("BlackBox2.h5")
model.summary()
model.predict(X_test , verbose = 1)
