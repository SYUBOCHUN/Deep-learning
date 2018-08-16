# 2017/9/20 SimpleRNN
import numpy as np
import pandas
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(1337)
t=0
LR = 0.001

training_dataframe = pandas.read_csv(r"C:\data\dengue\normalization\2016\kasanmin2016.csv", header=None) #training_dataset
testing_dataframe = pandas.read_csv(r"C:\data\dengue\normalization\2016\ntpsanchong2016.csv", header=None) #testing_dataset
training_dataset = training_dataframe.values
testing_dataset = testing_dataframe.values


x_train = training_dataset[:,0:5].astype(float)
y_train = training_dataset[:,5]
x_test = testing_dataset[:,0:5].astype(float)
y_test = testing_dataset[:,5]

x_train = x_train.reshape(-1, 1, 5)     # normalize
x_test = x_test.reshape(-1, 1, 5)      # normalize

def baseline_model():
# build RNN model
    model = Sequential()

# RNN cell
    model.add(SimpleRNN(batch_input_shape=(None, 1, 5),units=128))
    model.add(Dense(64,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax')) # 分多少類別

# optimizer
    adam = Adam(LR)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=2000, batch_size=50, verbose=2)
history = estimator.fit(x_train, y_train)
pre = estimator.predict(x_test)
print(pre)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('Accuracy Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()

for i in range(len(pre)):
    if pre[i] == y_test[i]:
        t = t+1
print((t/len(pre))*100)
