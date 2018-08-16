# 2017/9/20 CNN
import numpy as np
import pandas
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Embedding
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(1337)
t=0
LR = 0.001

training_dataframe = pandas.read_csv(r"C:\data\dengue\training_data.csv", header=None) #training_dataset
testing_dataframe = pandas.read_csv(r"C:\data\dengue\testing_data.csv", header=None) #testing_dataset
training_dataset = training_dataframe.values
testing_dataset = testing_dataframe.values


x_train = training_dataset[:,0:1024].astype('float32')
print(x_train.shape)
y_train = training_dataset[:,1026]
print(y_train.shape)

x_test = testing_dataset[:,0:1024].astype('float32')
y_test = testing_dataset[:,1026]

y_test_label = testing_dataset[:,1026]

# normalize
x_train = x_train.reshape(526,32,32,1)
x_test = x_test.reshape(526,32,32,1) 

y_train = y_train.reshape(263,2)
y_test = y_test.reshape(263,2)

# train
print("train data:",x_train.shape)
print("train labels:",y_train.shape)
# test
print("test data:",x_test.shape)
print("test labels:",y_test.shape)

# label do One-hot encoding 
y_train= np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def baseline_model():
# build RNN model
    model = Sequential()

# CNN cell
    model.add(Conv2D(filters=8, kernel_size=(3,3),
                     input_shape = (32, 32, 1),
                     activation='sigmoid',
                     padding='same'))
    model.add(Dropout(0.1))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    
    model.add(Dense(130, activation='sigmoid'))
    model.add(Dropout(rate=0.25))
    
    model.add(Dense(2, activation='softmax'))
# optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=2000, verbose=2)
history = estimator.fit(x_test, y_test,
                        batch_size = 150)
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
    if pre[i] == y_test_label[i]:
        t = t+1
print((t/len(pre))*100)

