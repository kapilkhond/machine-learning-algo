import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

# Load the boston dataset
bostonDataset = datasets.load_boston()

bostonData=(bostonDataset.data)
#adding column 0 with 1 value
bostonData=np.insert(bostonData,0,1,axis=1)

#train data x
trainX=bostonData[:-10]
#train data y
trainY=(bostonDataset.target[:-10])
#test data x
testX=bostonData[-10:]
#print testX.shape


#test data y
testY=bostonDataset.target[-10:]

m=trainX.shape[0]
n=trainX.shape[1]
theta=np.full(shape=(n,1),fill_value=1,dtype=float)
#print theta

trainingSize=m
featureNo=n
alpha=0.000005
prevcost = 1e9
maxIter = 1000000
currIter = 0

def hypothesis(theta):
	#print theta
	hyp=[]
	for i in trainX:
		product=[]
		product=np.dot(i,theta)
		#print product
		hyp.append(product)
		
	hyp=np.array(hyp)
	hypY=hyp.reshape(m,)
	return hypY

def compute_cost(hyp_Y):
	cost=0.0
	cost=(((hyp_Y-trainY)**2).sum())/(2.0*m)	
	return cost 


def gradient_descent(hyp_Y,theta):
	#print "before"
	#print theta
	for i in range(n):

		inter=((hyp_Y-trainY)*trainX[:,i]).sum()
		theta[i]=theta[i]-inter*(1.0/m)*alpha
		
	#print "after"
	#print theta		
	return theta	
		
while True:
	currIter += 1
	
	print currIter

	#print theta
	if np.isnan(np.min(theta)):
		break
			
	hyp_Y=hypothesis(theta)
	cost=compute_cost(hyp_Y)
	if prevcost<cost:
		print prevcost,cost
		break
	if currIter < maxIter:
		theta=gradient_descent(hyp_Y,theta)
		#print theta
		prevcost=cost
		print cost
		print theta
	else:
		break	


testOutcome=np.dot(testX,theta)
print testOutcome
print testY

'''
costCheck = 0.0

for x,y in zip(diabetes_train_X,diabetes_train_Y):
	#print x*theta1 + theta0, ' -- ', y
	costCheck += (x*theta1 + theta0 - y) **2

#print theta0, theta1, cost, costCheck

finalHyp=[]
for entry in diabetes_train_X:
	finalHyp.append(entry*theta1+theta0)


for eachTestX,eachTestY in zip(diabetes_test_X,diabetes_test_Y): 
	print theta1*eachTestX+theta0,eachTestY,"----",theta1*eachTestX+theta0-eachTestY
'''