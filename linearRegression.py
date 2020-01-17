import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()



# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X3 = diabetes_X.tolist()
diabetes_Y2 = diabetes.target.tolist()

diabetes_X2 = []

for i in diabetes_X3:
	diabetes_X2.append(i[0])

print len(diabetes_X2)
print len(diabetes_Y2)

'''
plt.plot(diabetes_X2,diabetes_Y2,'ro')
plt.title('diabetes')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()	
'''

diabetes_train_X=diabetes_X2[:-20]
diabetes_train_Y=diabetes_Y2[:-20]

diabetes_test_X=diabetes_X2[-20:]
diabetes_test_Y=diabetes_Y2[-20:]
print len(diabetes_test_X)
print len(diabetes_test_Y)


trainingSize=len(diabetes_train_X)
theta0=150
theta1=1000
alpha=0.5
prevcost = 1e9
maxIter = 1000
currIter = 0

def hypothesis(theta0,theta1):
	hyp_Y=[]
	for i in diabetes_train_X:
		hyp_Y.append(theta0+theta1*i)
	return hyp_Y
	#print hyp_Y

def compute_cost(hyp_Y):
	cost=0.0
	for hyp,actual in zip(hyp_Y,diabetes_train_Y):
		cost+=(hyp-actual)*(hyp-actual)
	cost=cost/float(2*trainingSize)
	print cost
	return cost	

def gradient_descent(hyp_Y,theta0,theta1):
	costThetaZero=0.0
	costThetaOne=0.0
	for hyp,actual,x in zip(hyp_Y,diabetes_train_Y,diabetes_train_X):
		costThetaZero+=hyp-actual
	   	costThetaOne+=(hyp-actual)*x
	#print costThetaZero
	#print costThetaOne
	theta0=theta0-alpha*float(1.0/trainingSize)*costThetaZero
	theta1=theta1-alpha*float(1.0/trainingSize)*costThetaOne
	return (theta0,theta1)


while True:
	currIter += 1
	hyp_Y=hypothesis(theta0,theta1)

	cost=compute_cost(hyp_Y)

	if currIter < maxIter:
		tupleTheta=gradient_descent(hyp_Y,theta0,theta1)
		#print type(tupleTheta)
		theta0=tupleTheta[0]
		theta1=tupleTheta[1]
		prevcost = cost
	else:
		break

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
print type(diabetes_train_X)
print type(diabetes_train_Y)


fig=plt.figure()
plt.plot(diabetes_train_X,diabetes_train_Y,'ro')
plt.plot(diabetes_train_X,finalHyp,color='blue',linewidth=3)
plt.title('diabetes')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()
plt.close("all")


# hyp_Y=hypothesis(theta0,theta1)
# cost=compute_cost(hyp_Y)
# print cost


'''
#print theta0
#print theta1
hyp_Y2=[]
for i in diabetes_train_X:
	hyp_Y2.append(theta0+theta1*i)

#print hyp_Y2
costFunction=0.0
for hyp,actual in zip(hyp_Y2,diabetes_train_Y):
	costFunction+=(hyp-actual)*(hyp-actual)
print costFunction
'''
# Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# print len(diabetes_X_test)
