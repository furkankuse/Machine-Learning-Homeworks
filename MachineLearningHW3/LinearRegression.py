import numpy as np


class LinearRegression:

    def __init__(self, trainingData, testData, k):
        self._trainingData = trainingData
        self._testData = testData
        self._w = []
        self._wk = []
        self._k = k

    def createX(self, dataSet):
        X = []

        for i in range(len(dataSet)):
            X.append(
                [1, dataSet[i][0], dataSet[i][1], dataSet[i][0] ** 2, dataSet[i][1] ** 2, dataSet[i][0] * dataSet[i][1],
                 np.absolute(dataSet[i][0] - dataSet[i][1]), np.absolute(dataSet[i][0] + dataSet[i][1])])

        return X

    def createY(self, dataSet):
        Y = []

        for i in range(len(dataSet)):
            Y.append(dataSet[i][2])

        return Y

    def train(self, choice):
        if choice:  # This part is for weight decay
            X = np.array(self.createX(self._trainingData))
            Y = np.array(self.createY(self._trainingData))
            I = np.identity(len(X[0]))
            pinv = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(10 ** self._k, I)), X.T)
            self._w = np.dot(pinv, Y)
        else:  # This part is linear regression wo weight decay
            pinv = np.linalg.pinv(self.createX(self._trainingData))
            Y = self.createY(self._trainingData)
            self._w = np.dot(pinv, Y)
        return self.calcEin(), self.caclEout(), self._w

    def calcEin(self):
        hit = 0
        X = self.createX(self._trainingData)
        for i in range(len(self._trainingData)):
            predict = self.calcYsign(self._w, X[i])
            if predict == self._trainingData[i][2]:
                hit += 1

        return 1 - (hit / len(self._trainingData))

    def caclEout(self):
        hit = 0
        X = self.createX(self._testData)
        for i in range(len(self._testData)):
            predict = self.calcYsign(self._w, X[i])
            if predict == self._testData[i][2]:
                hit += 1

        return 1 - (hit / len(self._testData))

    def calcYsign(self, w, X):
        value = 0
        for i in range(len(w)):
            value += w[i] * X[i]

        return np.sign(value)
