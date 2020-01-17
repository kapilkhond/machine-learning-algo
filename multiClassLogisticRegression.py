import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
start_time = time.time()
# import iris data to play with


iris = datasets.load_iris()
initialData=iris.data    # taking first two features
shape=initialData.shape
datapoints=shape[0]
features=shape[1]
aInput=initialData[:50]
bInput=initialData[50:100]
cInput=initialData[100:]


#data=initialData[0:130]
aTrain=aInput[0:40]
aTest=aInput[40:]
bTrain=bInput[:40]
bTest=bInput[40:]
cTrain=cInput[:40]
cTest=cInput[40:]

#np.ndarray(())
data=np.concatenate((aTrain,bTrain,cTrain))
data=np.insert(data,0,1,axis=1)
shape=data.shape
m=shape[0]
n=shape[1]
#print data

testData=np.concatenate((aTest,bTest,cTest))
testData=np.insert(testData,0,1,axis=1)
#print testData

aOutput=iris.target[:50]
bOutput=iris.target[50:100]
cOutput=iris.target[100:]

aTrainTarget=aOutput[:40]
aTestTarget=aOutput[40:]
bTrainTarget=bOutput[:40]
bTestTarget=bOutput[40:]
cTrainTarget=cOutput[:40]
cTestTarget=cOutput[40:]

#target=np.concatenate((aTrainTarget,bTrainTarget,cTrainTarget))
target=np.ndarray((120,1),dtype=int)
target[:40]=0
target[40:]=1
# print target
# print target.shape

testTarget=np.concatenate((aTestTarget,bTestTarget,cTestTarget))
testTarget=np.ndarray((30,1),dtype=int)
testTarget[:10]=0
testTarget[10:]=1
print testTarget

#print testTarget


#print target


'''
output=np.ndarray((150,1),dtype=int)
#output[:100]=0 # case 2 
#output[100:]=1 #case 2

#output[:50]=0 # case 1 
#output[50:]=1 #case 1

output[:50]=0 # case 3 
output[50:100]=1 #case 3
output[100:150]=0 #case 3
'''
#target=output[0:130]
#print target

#testData=initialData[130:150]
#testData=np.insert(testData,0,1,axis=1)

#testTarget = output[130:150]
#print testTarget
#print target
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
while(i<10000): #no of iterations for gradient descent
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

product=np.dot(testData,theta)
hypTest=(1.0/(1.0+np.e**(-1.0*product)))

print hypTest

print("--- %s seconds ---" % (time.time() - start_time))

#plotting part
fig=plt.figure()
testData_1 = testData[:10]
testData_2 = testData[10:20]
testData_3=testData[20:]
x1Min=np.min(testData[:,1]) # minimum of first feature
x1Max=np.max(testData[:,1]) #maximum of first feature 
ex1 = np.linspace(0, 10, 100)
#print ex1
ex2 = -(theta[1,0]* ex1 + theta[0, 0]) / theta[2,0]
#print ex2
x2Min=np.min(ex2)  # minimum of second feature
x2Max=np.max(ex2)  #maximum of second feature
plt.plot(testData_1[:,1],testData_1[:,2],'ro') # plotting points
plt.plot(testData_2[:,1],testData_2[:,2],'bo') # plotting points 
plt.plot(testData_3[:,1],testData_3[:,2],'co')
plt.plot(ex1,ex2,color='cyan') # plotting decision boundary
plt.axis([0,10,0,10])
#plt.axis([x1Min-2,x1Max+2,x2Min-2,+x2Max+2])  #setting axis as per minimum and maximum of first and second feature
plt.show()
plt.xlabel('x1')
plt.ylabel('x2')
plt.close("all")



