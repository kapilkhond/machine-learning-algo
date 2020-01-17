import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
start_time = time.time()
# import iris data to play with


iris = datasets.load_iris()
#print (iris.data.shape)
print iris.target.shape
initialData=iris.data    # taking first two features
#print initialData
'''
data=initialData[0:130]         
data=np.insert(data,0,1,axis=1)
shape=data.shape
m=shape[0]
n=shape[1]

target=iris.target[0:130]		
target=target.reshape(m,1)

testData=initialData[130:150]
testData=np.insert(testData,0,1,axis=1)

testTarget=iris.data[130:150]


'''

data=initialData[50:90]          # taking first 40 as training set for input
target=iris.target[50:90]		# taking first 40 as training set for output

data=np.concatenate((data,initialData[100:140]))  # concat 50-90 as training set for input
target=np.concatenate((target,iris.target[100:140])) # concat 50-90 as training set for output
data=np.insert(data,0,1,axis=1)


testData=iris.data[90:100] # taking 40-50 as testing set for input
testData=np.concatenate((testData,initialData[140:150]))  # concat 90-100 as testing set for input
testData=np.insert(testData,0,1,axis=1)

shape=data.shape
m=shape[0]
n=shape[1]

testShape=testData.shape
target=target.reshape(m,1)
testTarget=(iris.target[90:100]) # taking 40-50 as testing set for output
testTarget=np.concatenate((testTarget,iris.target[140:150]))  # concat 90-100 as testing set for output
testTarget=testTarget.reshape(testShape[0],1)

target[:40]=0
target[40:]=1


testTarget[:10]=0
testTarget[10:]=1
#print testTarget




alpha=0.01

theta=np.full(shape=(n,1),fill_value=1,dtype=float) #initialise theta

def hypothesis(theta):
	product=np.dot(data,theta)
	hypY=(1.0/(1.0+np.e**(-1.0*product)))
	return hypY

def costFunction(hypY):
	logHypY=np.log(hypY)
	firstTerm=target*logHypY
	secondTerm=(1-target)*np.log(1-hypY)
	sum=(firstTerm+secondTerm).sum()
	cost=(-1.0/m)*(sum)
	return cost


def gradientDescent(hypY):
	diff=hypY-target
	for i in range(n):
		newData=data[:,i].reshape(m,1)
		arrA=(diff*newData).sum()
		theta[i,0]=theta[i,0]-(alpha*(arrA))
	return theta	
	

i=0
while(i<1000): #no of iterations for gradient descent
	hypY=hypothesis(theta)
	cost=costFunction(hypY)
	theta=gradientDescent(hypY)
	i=i+1

#for testing data

'''
#to calculate whether point lies above or below decision boundary
testTarget=testTarget.reshape(20,)
x1=testData[:,1]
x2=testData[:,2]
x2Calculated = -(theta[1,0]* x1 + theta[0, 0]) / theta[2,0]
for i,j,k in zip(x2,x2Calculated,testTarget):
	print i,j,k
'''
#print theta
product=np.dot(testData,theta)
#print product
hypTest=(1.0/(1.0+np.e**(-1.0*product)))

print hypTest
#print("--- %s seconds ---" % (time.time() - start_time))


