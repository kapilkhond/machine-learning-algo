import numpy as np

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)
m=X.shape[0]
#print X.shape

# Normalize
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

#X=np.insert(X,0,1,axis=1)

#print X
class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.hiddenLayerSize, (self.inputLayerSize))
        self.W2 = np.random.randn(self.outputLayerSize, (self.hiddenLayerSize))
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, np.transpose(self.W1))
        self.a2 = self.sigmoid(self.z2)
        #self.a2=np.insert(self.a2,0,1,axis=1)
        #print self.a2
        self.z3 = np.dot(self.a2, np.transpose(self.W2))
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def costFunction(self,hyp,y):
    	cost=(1.0/2*m)*(np.sum((hyp-y)**2))    
    	return cost

    def sigmoidDerivative(self,z):
    	#return ((np.exp(-z))/((1+np.exp(-z))**2))	
    	return np.exp(-z)/((1+np.exp(-z))**2)

    def gradientDescent(self,hyp,y):
    	temp1=(y-hyp)	
    	temp2=self.sigmoidDerivative(self.z3)
    	print temp2.shape
        temp3=np.multiply(-(temp1),temp2)
    	#print temp3.shape
    	#print self.a2.shape
    	djdw2=np.dot(np.transpose(self.a2),temp3)
        #print self.W2.shape
        #print djdw2
    	# print grad.shape
    	# print self.W2
    	# print temp3.shape
    	#print self.W2.shape
    	#print self.W2
    	#print grad
    	tempW2=(np.transpose(self.W2))-(0.01*djdw2)
    	#print tempW2
        #print temp3.shape
    	temp=np.dot(temp3,self.W2)
    	temp4=self.sigmoidDerivative(self.z2)
    	#print self.z2.shape
        #print self.z3
        #print temp.shape
    	#print temp4.shape
    	#print np.multiply(temp,temp4)




nn=Neural_Network()
hyp=nn.forward(X)
#print y
#print hyp.shape
cost=nn.costFunction(hyp,y)
#print cost
nn.gradientDescent(hyp,y)



