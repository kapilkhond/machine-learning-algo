import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn import datasets, linear_model



m=100
n=2
hackerrankData=genfromtxt('data.csv',delimiter=' ')
hackerrankDataCopy= hackerrankData.copy()
hackerrankDataTrainX=np.delete(hackerrankDataCopy,2,axis=1)
hackerrankDataTrainY=np.delete(hackerrankDataCopy,np.s_[:2:1],axis=1)

hackerrankDataTrainX=np.insert(hackerrankDataTrainX,0,1,axis=1)
#print hackerrankDataTrainX
#hackerrankDataTrainX=hackerrankDataTrainX*hackerrankDataTrainX
hackerrankDataTrainListX=hackerrankDataTrainX.tolist()
modList=[]
for entry in hackerrankDataTrainListX:
 	temp1=entry[1]
 	temp2=entry[2]
 	modList.append([entry[0],temp1,temp2,temp1*temp1,temp2*temp2,temp1*temp2]) 
hackerrankDataTrainX=np.array(modList)
temp=np.linalg.pinv(np.dot(hackerrankDataTrainX.transpose(),hackerrankDataTrainX))
theta=np.dot(np.dot(temp,hackerrankDataTrainX.transpose()),hackerrankDataTrainY)
hackerrankDataTestX=np.array([[0.05,0.54,0.0025,0.2916,0.027],[0.91,0.91,0.8281,0.8281,0.8281],[0.31,0.76,0.0961,0.5776,0.2356],[0.51,0.31,0.2601,0.0961,0.1581]])
hackerrankDataTestX=np.insert(hackerrankDataTestX,0,1,axis=1)
testXTranspose=hackerrankDataTestX.transpose()
hypY=np.dot(theta.transpose(),testXTranspose)
hackerrankDataTestY=np.array([[180.38,1312.07,440.13,343.72]])
cost=0.0
finalX=[insideEntry for entry in hypY for insideEntry in entry ]
finalY=[insideEntry for entry in hackerrankDataTestY for insideEntry in entry ]
print finalX
print finalY
for hyp,y in zip(finalX,finalY):
	cost+=(hyp-y)**2
cost=cost/(2.0*4)
print cost
print theta
'''
hackerrankData=genfromtxt('data.csv',delimiter=' ')
hackerrankDataCopy= hackerrankData.copy()
hackerrankDataTrainX=np.delete(hackerrankDataCopy,2,axis=1)
hackerrankDataTrainY=np.delete(hackerrankDataCopy,np.s_[:2:1],axis=1)
hackerrankDataTrainX=np.insert(hackerrankDataTrainX,0,1,axis=1)
hackerrankDataTrainListX=hackerrankDataTrainX.tolist()
modList=[]
for entry in hackerrankDataTrainListX:
 	temp1=entry[1]
 	temp2=entry[2]
 	modList.append([entry[0],temp1,temp2,temp1*temp1,temp2*temp2,temp1*temp2]) 
modArray=np.array(modList)
print modArray
'''
'''
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
inputParameters=raw_input()
inputParametersSplit=inputParameters.split(' ')
n=int(inputParametersSplit[0])
m=int(inputParametersSplit[1])
trainX=np.ndarray(shape=(m,n), dtype=float)
trainY=np.ndarray(shape=(m),dtype=float)
#print matrix
for i in range(m):
    splitvalue=(raw_input().split(' '))
    for j in range(n+1):
        if j<n:
            trainX[i][j]=float(splitvalue[j])
        else:
            trainY[i]=float(splitvalue[j])
#print trainY
trainX=np.insert(trainX,0,1,axis=1)
hackerrankDataTrainListX=trainX.tolist()
modList=[]
for entry in hackerrankDataTrainListX:
 	temp1=entry[1]
 	temp2=entry[2]
 	modList.append([entry[0],temp1,temp2,temp1*temp1,temp2*temp2,temp1*temp2]) 
trainX=np.array(modList)
temp=np.linalg.pinv(np.dot(trainX.transpose(),trainX))
theta=np.dot(np.dot(temp,trainX.transpose()),trainY)
#print theta
trainingSize=int(raw_input())
testX=np.ndarray(shape=(trainingSize,n),dtype=float)
for i in range(trainingSize):
    splitvalue=(raw_input().split(' '))
    for j in range(n):
        testX[i][j]=float(splitvalue[j])            
testX=np.insert(testX,0,1,axis=1)
hackerrankDataTestListX=testX.tolist()
modList=[]
for entry in hackerrankDataTestListX:
 	temp1=entry[1]
 	temp2=entry[2]
 	modList.append([entry[0],temp1,temp2,temp1*temp1,temp2*temp2,temp1*temp2]) 
testX=np.array(modList)
hypY=np.dot(theta,testX.transpose())

Y=[180.38,1312.07,440.13,343.72]
cost=0
for hypy ,y in zip(hypY,Y):
      cost+=(hypy-y)**2  
cost=cost/(2.0*4)
print cost

for entry in hypY:
	print entry

'''