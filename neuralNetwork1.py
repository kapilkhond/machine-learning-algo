import numpy as np

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
Y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X/np.amax(X, axis=0)
Y = Y/100 #Max test score is 100
class Neural_Network(object):
	def __init__(self):        
		#Define Hyperparameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		#Weights (parameters)
		self.W1 = np.random.randn(self.hiddenLayerSize, self.inputLayerSize+1)
		self.W2 = np.random.randn(self.outputLayerSize, self.hiddenLayerSize+1)
	
	def multiply(self,X):
		#print X.shape
		#print self.W1.shape
		X=np.insert(X,0,1,axis=1)
		self.Z2=np.dot(X,np.transpose(self.W1))
		self.A2=self.sigmoid(self.Z2)
		self.A2=np.insert(self.A2,0,1,axis=1)
		self.Z3=np.dot(self.A2,np.transpose(self.W2))
		self.hyp=self.sigmoid(self.Z3)
		return 0
	
	def sigmoid(self,Z):		
		A=(1.0/(1.0+np.exp(Z)))
		return A

	def costFunction(self):		
		cost=0.5*np.sum((Y-self.hyp)**2)	
		return cost

nn=Neural_Network()
nn.multiply(X)
print  nn.hyp
cost=nn.costFunction()
