import numpy as np
from random import shuffle


def prepareTrainInput(number, trainData):   # This one labels given number as +1 and rest -1
    x = []
    y = []
    for i in range(len(trainData)):
        x.append({0: trainData[i][1], 1: trainData[i][2]})

        if number == trainData[i][0]:
            y.append(1)
        else:
            y.append(-1)

    return y, x


def prepareForOneVsFive(data):  # This one extracts ones and fives from given data set, and labels 1 as 1, 5 as -1
    x = []
    y = []
    for i in range(len(data)):
        if data[i][0] == 1:
            x.append({0: data[i][1], 1: data[i][2]})
            y.append(1)

        elif data[i][0] == 5:
            x.append({0: data[i][1], 1: data[i][2]})
            y.append(-1)

    return y, x


def shuffleData(dataY, dataX):  # This one is used one Cross Validation, for shuffling data
    shuffled = np.array(range(len(dataY)))
    shuffle(shuffled)
    newX = []
    newY = []

    for i in range(len(shuffled)):
        newX.append(dataX[shuffled[i]])
        newY.append(dataY[shuffled[i]])

    return newY, newX


class Data:

    def __init__(self, trainName, TestName=None):  # For reading the data
        # Some of the problems use only trains set, so test is not mandatory
        self._trainingData = np.loadtxt(trainName, dtype=np.float64)
        if TestName is not None:
            self._testData = np.loadtxt(TestName, dtype=np.float64)
        else:
            self._testData = None

    @property
    def trainingData(self):
        return self._trainingData

    @property
    def testData(self):
        return self._testData
