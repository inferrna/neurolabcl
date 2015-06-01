from habranet import network
nt = network()
nn=nt.create([4, 1000, 1])

lamb=0.3
cost=1
alf = 0.2
xTrain = [[1, 2.3, 4.5, 5.3], [1.1, 1.3, 2.4, 2.4], [1.9, 1.7, 1.5, 1.3], [2.3, 2.9, 3.3, 4.9], [3, 5.2, 6.1, 8.2], [3.31, 2.9, 2.4, 1.5], [4.9, 5.7, 6.1, 6.3],
 [4.85, 5.0, 7.2, 8.1], [5.9, 5.3, 4.2, 3.3], [7.7, 5.4, 4.3, 3.9], [6.7, 5.3, 3.2, 1.4], [7.1, 8.6, 9.1, 9.9], [8.5, 7.4, 6.3, 4.1], [9.8, 5.3, 3.1, 2.9]]
yTrain = [[1], [1], [0], [1], [1], [0], [1],
 [1], [0], [0], [0], [1], [0], [0]]

xTest= [[0.4, 1.9, 2.5, 3.1], [1.51, 2.0, 2.4, 3.8], [2.6, 5.1, 6.2, 7.2], [3.23, 4.1, 4.3, 4.9], [7.1, 7.6, 8.2, 9.3],
 [5.78, 5.1, 4.5, 3.55], [6.33, 4.8, 3.4, 2.5], [7.67, 6.45, 5.8, 4.31], [8.22, 6.32, 5.87, 3.59], [9.1, 8.5, 7.7, 6.1]]
yTest = [[1], [1], [1], [1], [1],
 [0], [0], [0], [0], [0]]
                
while cost>0:
    cost=nt.costTotal(False, nn, xTrain, yTrain, lamb)
    costTest=nt.costTotal(False, nn, xTest, yTest, lamb)
    delta=nt.backpropagation(False, nn, xTrain, yTrain, lamb)
    nn['theta']=[nn['theta'][i]-alf*delta[i] for i in range(0,len(nn['theta']))]
    print('Train cost ', cost, 'Test cost ', costTest)
    print(nt.runAll(nn, xTest))
