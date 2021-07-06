import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('./fer2013.csv')

width, height = 48, 48

datapoints = data['pixels'].tolist()

X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))


X = np.asarray(X)
X = np.expand_dims(X, -1)

y = pd.get_dummies(data['emotion']).as_matrix()

np.save('fdataX', X)
np.save('flabels', y)

print("Preprocessing Done")
print("Number of features: "+str(len(X[0])))
print("Number of labels: "+str(len(y[0])))
print("Number of examples in dataset: "+str(len(X)))
print("X,y stored in fdataX.npy and flabels.npy respectively")