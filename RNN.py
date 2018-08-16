#==/Create first network with Keras/====================================
from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt
t = 0
#==/Fix random seed for reproducibility/================================
numpy.random.seed(7)
#==/Load pima indians dataset/==========================================
dataset = numpy.loadtxt(r"C:\\SNNS\tpshilin2014.csv", delimiter=",")
dataset2 = numpy.loadtxt(r"C:\\SNNS\kasanmin2014.csv", delimiter=",")
#==/Split into input (X) and output (Y) variables/======================
X = dataset[:,1:10]
X2 = dataset2[:,1:10]
Y = dataset[:,10]
Y2 = dataset2[:,10]

#==/Create model/=======================================================
model = Sequential()
model.add(Dense(12, input_dim=9, init='uniform', activation='sigmoid'))
model.add(Dense(8, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
#==/Compile model/======================================================
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
#==/Fit the model/======================================================

history = model.fit(X, Y, nb_epoch=1000, batch_size=10, verbose=2)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#==/Calculate predictions/==============================================
predictions = model.predict(X2)
#==/Round predictions/==================================================
rounded = [round(x[0]) for x in predictions]
print(rounded)


#==/Accuracy/=======================================================
for i in range(len(predictions)):
    if rounded[i] == Y2[i]:
        t = t+1
print((t/len(predictions))*100) 
