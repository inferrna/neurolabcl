import copy
import numpy as np
import random as rd

class network:
    # layers -list [5 10 10 5] - 5 input, 2 hidden layers (10 neurons each), 5 output
    def create(self, layers):
        theta=[0]
        for i in range(1, len(layers)): # for each layer from the first (skip zero layer!)
            theta.append(np.random.uniform(-1, 1, (layers[i], layers[i-1]+1))) # create nxM+1 matrix (+bias!) with random floats in range [-1; 1]
        nn={'theta':theta,'structure':layers}
        return nn
    
    def runAll(self, nn, X):
        z=[0]
        m = len(X)
        a = [ copy.deepcopy(X) ] # a[0] is equal to the first input values
        logFunc = self.logisticFunction()
        for i in range(1, len(nn['structure'])): # for each layer except the input
            a[i-1] = np.concatenate((np.ones((m,1,)), a[i-1]), axis=1); # add bias column to the previous matrix of activation functions
            z.append(np.dot(a[i-1], nn['theta'][i].T)) # for all neurons in current layer multiply corresponds neurons
            #print("Shapes is ", a[i-1].shape, nn['theta'][i].T.shape)
            #print("Result is ", z[-1].shape)
            # in previous layers by the appropriate weights and sum the productions
            a.append(logFunc(z[i])) # apply activation function for each value
        nn['z'] = z
        nn['a'] = a
        return a[len(nn['structure'])-1]
        
    def run(self, nn, input):
        z=[0]
        a=[]
        a.append(copy.deepcopy(input))
        a[0]=np.array(a[0]).T # nx1 vector
        logFunc = self.logisticFunction()
        for i in range(1, len(nn['structure'])):
            a[i-1]=np.vstack(([1], a[i-1]))
            z.append(np.dot(nn['theta'][i], a[i-1]))
            a.append(logFunc(z[i]))
        nn['z'] = z
        nn['a'] = a
        return a[len(nn['structure'])-1]
        
    def logisticFunction(self):
        return lambda x: 1/(1+np.exp(-x))
        
    def costTotal(self, theta, nn, X, y, lamb):
        m = len(X)
        #following string is for fmin_cg computaton
        if type(theta) == np.ndarray: nn['theta'] = self.roll(theta, nn['structure'])
        y = np.array(copy.deepcopy(y))
        hAll = self.runAll(nn, X) #feed forward to obtain output of neural network
        cost = self.cost(hAll, y)
        return cost/m+(lamb/(2*m))*self.regul(nn['theta']) #apply regularization 
    
    def cost(self, h, y):
        logH=np.log(h)
        log1H=np.log(1-h)
        cost = np.dot(-1*y.T, logH) - np.dot((1-y.T), log1H) #transpose y for matrix multiplication
        return cost.sum() # sum matrix of costs for each output neuron and input vector
        
    def regul(self, theta):
        reg=0
        thetaLocal=copy.deepcopy(theta)
        for i in range(1,len(thetaLocal)):
            thetaLocal[i]=np.delete(thetaLocal[i],0,1) # delete bias connection
            thetaLocal[i]=np.power(thetaLocal[i], 2) # square the values because they can be negative
            reg+=thetaLocal[i].sum() # sum at first rows, than columns
        return reg
        
    def backpropagation(self, theta, nn, X, y, lamb):
        layersNumb=len(nn['structure'])
        thetaDelta = [0]*(layersNumb)
        m=len(X)
        #calculate matrix of outpit values for all input vectors X
        hLoc = copy.deepcopy(self.runAll(nn, X))
        yLoc = np.array(y)
        thetaLoc = copy.deepcopy(nn['theta'])
        derFunct=np.vectorize(lambda x: (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x)))) )
        
        zLoc = copy.deepcopy(nn['z'])
        aLoc = copy.deepcopy(nn['a'])
        for n in range(0, len(X)):
            delta = [0]*(layersNumb+1)  #fill list with zeros
            delta[len(delta)-1] = (hLoc[n].T-yLoc[n].T) #calculate delta of error of output layer
            delta[len(delta)-1] = delta[len(delta)-1].reshape(1, -1)
            for i in range(layersNumb-1, 0, -1):
                if i>1: # we can not calculate delta[0] because we don't have theta[0] (and even we don't need it)
                    z = zLoc[i-1][n]
                    z = np.c_[ [[1]], z.reshape((1,)*(2-z.ndim) + z.shape) ] #add one for correct matrix multiplication
                    delta[i]=np.multiply(np.dot(thetaLoc[i].T, delta[i+1]).reshape(-1, 1), derFunct(z).T)
                    delta[i]=delta[i][1:]
                #print(thetaDelta[i], delta[i+1].shape, aLoc[i-1][n], '\n')
                #print(np.dot(thetaLoc[i].T, delta[i+1]).shape, derFunct(z).T.shape, '\n')
                #print(delta[i+1].shape, aLoc[i-1][n].shape )
                thetaDelta[i] = thetaDelta[i] + np.dot(delta[i+1].reshape(-1, 1), aLoc[i-1][n].reshape(1, -1)) #delta[i+1]*aLoc[i-1][n]
                #exit()

        for i in range(1, len(thetaDelta)):
            thetaDelta[i]=thetaDelta[i]/m
            thetaDelta[i][:,1:]=thetaDelta[i][:,1:]+thetaLoc[i][:,1:]*(lamb/m) #regularization
       
        if type(theta) == np.ndarray: return np.asarray(self.unroll(thetaDelta)).reshape(-1) # to work also with fmin_cg
        return thetaDelta
    
    # create 1d array form lists like theta
    def unroll(self, arr):
        for i in range(0,len(arr)):
            arr[i]=np.array(arr[i])
            if i==0:
                res=(arr[i]).ravel().T
            else:
                res=np.vstack((res,(arr[i]).ravel().T))
        res.shape=(1, len(res))
        return res
    # roll back 1d array to list with matrices according to given structure
    def roll(self, arr, structure):
        rolled=[arr[0]]
        shift=1
        for i in range(1,len(structure)):
            print(type(structure[i]), " * ", type(structure[i-1]+1))
            exit()
            temparr=copy.deepcopy(arr[shift:shift+structure[i]*(structure[i-1]+1)])
            temparr.shape=(structure[i],structure[i-1]+1)
            rolled.append(np.array(temparr)) #NEED COMPARE WITH MATRIX
            print(rolled[-1].shape)
            exit() #DEBUG NOT FIRED
            shift+=structure[i]*(structure[i-1]+1)
        return rolled
