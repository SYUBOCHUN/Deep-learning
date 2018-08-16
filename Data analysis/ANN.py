import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#==/Fix Random Seed for Reproducibility/===========================================================
seed = 7
numpy.random.seed(seed)
t = 0

#==/Load Dataset/==================================================================================
training_dataframe = pandas.read_csv("C:\\SNNS\KSF.csv", header=None) #training_dataset
testing_dataframe = pandas.read_csv("C:\\SNNS\KSF.csv", header=None) #testing_dataset
training_dataset = training_dataframe.values
testing_dataset = testing_dataframe.values

X = training_dataset[:,1:5].astype(float)
Y = training_dataset[:,5]
X2 = testing_dataset[:,1:5].astype(float)
Y2 = testing_dataset[:,5]

#==/Encode Class Values as Integers/===============================================================
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
#==/Convert Integers to Dummy Variables/===========================================================
dummy_y = np_utils.to_categorical(encoded_Y)

#==/Baseline Model/================================================================================
def baseline_model():
#==/Create Model/==================================================================================
	model = Sequential()
	model.add(Dense(1, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(512, kernel_initializer='normal', activation='relu'))
	model.add(Dense(256, kernel_initializer='normal', activation='relu'))
	model.add(Dense(128, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(2, kernel_initializer='normal', activation='softmax'))
#==/Compile Model/=================================================================================
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model
#==/Result Print/==============================================================================================
estimator = KerasClassifier(build_fn=baseline_model, epochs=2000, batch_size=10, verbose=2)
estimator.fit(X, Y)
predictions = estimator.predict(X)
print(predictions)

for i in range(len(predictions)):
    if predictions[i] == Y2[i]:
        t = t+1
print((t/len(predictions))*100)

