import random


numLayers = 10             #hidden and output layers (not inp)
nodes = 5                  #for every layer


weights = [[[(random.randint(1, 9)/100) for k in range(nodes)] for j in range(nodes)] for i in range(numLayers)]
biases = [[(random.randint(1, 9)/10) for j in range(nodes)] for i in range(numLayers)]


trainingData =  [[[4.70, 5.74, 7.01, 8.56, 0], [0, 0, 0, 0, 0]],         #CURRENTLY: trying to overfit
                 [[1, 0, 0, 0, 0], [7, 7, 7, 7, 7]],                     #TRY: three points on expo, quad, log graphs and classify them
                 [[2, 0, 0, 0, 0], [4, 4, 4, 4, 4]],
                 [[3, 0, 0, 0, 0], [4, 4, 4, 4, 4]]]

def relu(x):
    if(x >= 0):
        return x
    else:
        return 0.001*x



def reluDash(x):
    if(x >= 0):
        return 1
    else:
        return 0.01

def calculateOutputs(inputs, weights, biases, numLayers, nodes):
    values = [[0.0 for j in range(nodes)] for i in range(numLayers+1)]
    zeds = [[0.0 for j in range(nodes)] for i in range(numLayers+1)]

    totalWeightProducts = 0
    values[0] = inputs
    zeds[0] = inputs
    for layers in range (0, numLayers):
        for nextRows in range (0, nodes):
            for currentRows in range (0, nodes):
                totalWeightProducts += weights[layers][currentRows][nextRows]*values[layers][currentRows]  #no weights[layers+1] because it starts at the second layer!
            zeds[layers+1][nextRows] = totalWeightProducts + biases[layers][nextRows]
            values[layers+1][nextRows] = relu(zeds[layers+1][nextRows])
            totalWeightProducts = 0
    return values, zeds


def calculateGradients(values, zeds, y, weights, numLayers, nodes):
    weightGrads = [[[0 for k in range(nodes)] for j in range(nodes)] for i in range(numLayers)]

    biasGrads = [[0 for j in range(nodes)] for i in range(numLayers)]                             
    
    nodeGrads = [[0 for j in range(nodes)] for i in range(numLayers - 1)]       #only hidden layers

    
    for layersBack in range (0, numLayers):
        if(layersBack == 0):       #special case: finding grads using cost
            for preLayer in range(0, nodes):
                for currentLayer in range(0, nodes):
                    actFuncDiff = reluDash(zeds[numLayers][currentLayer])
                    costDiff = 2*(values[numLayers][currentLayer] - y[currentLayer])
                    
                    weightGrads[numLayers-1][preLayer][currentLayer] += values[numLayers-1][preLayer]*actFuncDiff*costDiff               #calculates weights
                    nodeGrads[numLayers-2][preLayer] += weights[numLayers-1][preLayer][currentLayer]*actFuncDiff*costDiff                #calculated wanted nodes (for later)
                    if(preLayer == 0):
                        biasGrads[numLayers-1][currentLayer] += costDiff*actFuncDiff                                                     #try taking the + out
        else:
            for preLayer in range(0, nodes):
                for currentLayer in range(0, nodes):
                    actFuncDiff = reluDash(zeds[numLayers - layersBack][currentLayer])
                    weightGrads[(numLayers-1) - layersBack][preLayer][currentLayer] += nodeGrads[(numLayers-1) - layersBack][currentLayer]*actFuncDiff*values[(numLayers-1) - layersBack][preLayer]  #calculate nodeGrads for more hidden layers
                    biasGrads[(numLayers-1) - layersBack][preLayer] += nodeGrads[(numLayers-1) - layersBack][currentLayer]*actFuncDiff
                    if(layersBack != (numLayers-1)):
                        nodeGrads[(numLayers-2) - layersBack][preLayer] += nodeGrads[(numLayers-1) - layersBack][currentLayer]*actFuncDiff*weights[(numLayers-1) - layersBack][preLayer][currentLayer]            
                    
    return weightGrads, biasGrads
        

#HYPERPARAMS
learningRate = 0.00005
epochs = 20000

#training
totalCost = 0
outputs = [0 for i in range(nodes)]
#decreasing = True
#costCompare = [0, 100]
for epoch in range(epochs):                    #keeps going until cost increases
    for data in trainingData:
        #calculates outputs from training data given the current model
        values, zeds = calculateOutputs(data[0], weights, biases, numLayers, nodes)     #data[0] are the features, data[1] are the labels
        outputs = values[numLayers]
        
        #calculates costs
        for i in range(0, nodes):
            totalCost += (outputs[i] - data[1][i])*(outputs[i] - data[1][i])

        #calculates gradients
        weightGrads, biasGrads = calculateGradients(values, zeds, data[1], weights, numLayers, nodes)


        #implement gradients
        for layers in range(0, numLayers):
            for preLayer in range (0, nodes):
                for currentLayer in range (0, nodes):
                    weights[layers][preLayer][currentLayer] -=  learningRate*weightGrads[layers][preLayer][currentLayer]
                biases[layers][preLayer] -= learningRate*biasGrads[layers][preLayer]

    print('EPOCH '+str(epoch)+', COST: '+str(totalCost))
    totalCost = 0

    #switch the cost compares
    """
    temp = costCompare[1]
    costCompare[1] = totalCost/(3*7)
    costCompare[0] = temp
    
    epoch += 1
    if(costCompare[1] >= costCompare[0] or epoch >= maxEpochs):
        decreasing = False
    """
       



#testing
testInps = [0 for i in range(nodes)]
cont = 'y'
while(cont != 'n'):
    for i in range(nodes):
        testInps[i] = float(input('test '+str(i+1)+': '))
    testVals, testZeds = calculateOutputs(testInps, weights, biases, numLayers, nodes)

    for i in range(nodes):
        print(str(testVals[numLayers][i]))
    cont = input('continue? ')
