import random
import numpy as np


class PLA:

    def __init__(self, trainingData, w):
        self._trainingData = trainingData
        self._trainingSet = trainingData.trainingSet
        self._testSet = trainingData.testSet
        self._w = w

    def train(self):

        misclassified = self._trainingSet.copy()
        iterations = 0
        while len(misclassified) > 0:
            index = random.randint(0, len(misclassified) - 1)
            self._w = self.updateW(self._w, misclassified[index])
            misclassified = self.updateData(self._w)
            iterations += 1

        return iterations, self._w

    def updateW(self, w, point):
        Xn = [1, point[0], point[1]]
        y = point[2]

        return w + np.dot(Xn, y)

    def updateData(self, w):
        misclassified = []

        for i in range(len(self._trainingSet)):
            self._trainingSet[i][3] = self._trainingData.calcYsign(w, self._trainingSet[i])
            if self._trainingSet[i][3] != self._trainingSet[i][2]:
                misclassified.append(self._trainingSet[i])

        return misclassified

    def calcProb(self):
        hit = 0
        miss = 0

        for i in range(len(self._testSet)):
            predict = self._trainingData.calcYsign(self._w, self._testSet[i])
            if predict == self._testSet[i][2]:
                hit += 1
            else:
                miss += 1

        return miss / len(self._testSet)
