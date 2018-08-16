import pandas
import numpy
import csv  
#==/Load Dataset/==================================================================================
training_dataframe = pandas.read_csv(r"C:\\data\dengue\2014\ntpsanchong2014-1.csv", header=None) #training_dataset
training_dataset = training_dataframe.values
a = training_dataset[:,0].astype(float)
b = training_dataset[:,1].astype(float)
c = training_dataset[:,2].astype(float)
d = training_dataset[:,3].astype(float)
e = training_dataset[:,4].astype(float)

def Normalization(x):
    return[(float(i)-min(x))/float(max(x)-min(x))for i in x]


a1 = a.astype(list)
b1 = b.astype(list)
c1 = c.astype(list)
d1 = d.astype(list)
e1 = e.astype(list)

a2 = Normalization(a1)
b2 = Normalization(b1)
c2 = Normalization(c1)
d2 = Normalization(d1)
e2 = Normalization(e1)


file = open(r"C:\\Users\user\Desktop\exportExample.csv", 'w',newline='')
writer = csv.writer(file)
csvHeader=zip(a2,b2,c2,d2,e2)
for row in csvHeader:
    writer.writerow(row)


file.close()

