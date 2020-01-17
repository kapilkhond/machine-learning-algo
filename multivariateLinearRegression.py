import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# Load the boston dataset
bostonDataset = datasets.load_boston()

#print bostonDataset.keys()

bostonData=(bostonDataset.data)
bostonData=np.insert(bostonData,0,1,axis=1)
#print bostonData
trainX=bostonData[:-10]

trainY=(bostonDataset.target[:-10])

testX=bostonData[-10:]
testY=bostonDataset.target[-10:]

temp=np.linalg.pinv(np.dot(trainX.transpose(),trainX))
theta=np.dot(np.dot(temp,trainX.transpose()),trainY)
print theta
testXTranspose= testX.transpose()
#print testXTranspose.shape
hypY=np.dot(theta,testXTranspose)

print hypY
print testY
cost=0
for hyp,y in zip(hypY,testY):
	cost+=(hyp-y)**2
cost/=(2.0*len(hypY))
print cost

