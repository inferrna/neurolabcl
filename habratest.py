from habranet import network
import numpy as np
nt = network()

lamb=0.3
cost=1
alf = 0.2
s = 10

data = np.load('/tmp/data.txt.npy')
res = np.load('/tmp/res.txt.npy')
nn=nt.create([data[0].size, data[0].size//2, data[0].size//3, res[0].size])

xTrain = data[:-s]
yTrain = res[:-s]

xTest = data[-s:]
yTest = res[-s:]
                
while cost:
    cost=nt.costTotal(False, nn, xTrain, yTrain, lamb)
    costTest=nt.costTotal(False, nn, xTest, yTest, lamb)
    delta=nt.backpropagation(False, nn, xTrain, yTrain, lamb)
    nn['theta']=[nn['theta'][i]-alf*delta[i] for i in range(0,len(nn['theta']))]
    print('Train cost ', cost, 'Test cost ', costTest)
    print(nt.runAll(nn, xTest))
