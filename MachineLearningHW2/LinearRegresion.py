import numpy as np

class LinearRegression:

    def __init__(self, trainingData, w, choice):
        self._trainingData = trainingData
        self._trainingSet = trainingData.trainingSet
        self._testSet = trainingData.testSet
        self._w = w
        self._choice = choice

    def createX(self):
        X = []
        for i in range(len(self._trainingSet)):
            ########## This line is used in linear data part
            if self._choice:
                newRow = [1, self._trainingSet[i][0], self._trainingSet[i][1]]
            ##########

            ########## This line is used in nonlinear data part (questions 12 and 13)
            else:
                newRow = [1, self._trainingSet[i][0], self._trainingSet[i][1], self._trainingSet[i][0] * self._trainingSet[i][1], self._trainingSet[i][0] ** 2, self._trainingSet[i][1] ** 2]
            ##########
            X.append(newRow)

        return X

    def createY(self):
        Y = []
        for i in range(len(self._trainingSet)):
            value = self._trainingSet[i][2]
            Y.append(value)

        return Y

    def findW(self):
        pinv = np.linalg.pinv(self.createX())
        Y = self.createY()
        return np.dot(pinv, Y)


    def train(self):
        self._w = self.findW()
        return len(self.updateData(self._w))

    def updateData(self, w):
        misclassified = []

        for i in range(len(self._trainingSet)):
            ########## This line is used in linear data part
            if self._choice:
                self._trainingSet[i][3] = self._trainingData.calcYsign(w, self._trainingSet[i])
            ##########

            ########## This two line are used in nonlinear part (quesitons 12 and 13)
            else:
                point = [self._trainingSet[i][0], self._trainingSet[i][1], self._trainingSet[i][0] * self._trainingSet[i][1], self._trainingSet[i][0] ** 2, self._trainingSet[i][1] ** 2]
                self._trainingSet[i][3] = self._trainingData.calcYsign(w, point)
            ##########
            if self._trainingSet[i][3] != self._trainingSet[i][2]:
                misclassified.append(self._trainingSet[i])

        return misclassified

    def calcEout(self):
        hit = 0
        miss = 0

        for i in range(len(self._testSet)):
            predict = self._trainingData.calcYsign(self._w, self._testSet[i])
            if predict == self._testSet[i][2]:
                hit += 1
            else:
                miss += 1

        return miss / len(self._testSet)

    @property
    def w(self):
        return self._w

