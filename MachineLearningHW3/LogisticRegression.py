import numpy as np
from math import e
import math
import random


class LogisticRegression:
    def __init__(self, n, dataset, w):
        self._n = n
        self._dataSet = dataset
        self._trainingSet = dataset.trainingSet
        self._testSet = dataset.testSet
        self._w = w

    def calcNorm(self, vector):
        total = 0
        for i in range(len(vector)):
            total += vector[i] ** 2

        return total ** (1 / 2)

    def train(self):
        wt = self._w
        difference = 1
        stepSize = 0
        wt1 = wt
        while difference >= .01:
            length = len(self._trainingSet)
            shuffled = np.array(range(length))
            random.shuffle(shuffled)
            for j in range(length):
                i = shuffled[j]
                X = [1, self._trainingSet[i][0], self._trainingSet[i][1]]
                y = self._trainingSet[i][2]
                grad = np.dot(-y, X) / (1 + e ** (y * np.dot(wt1, X)))

                wt1 = wt1 - grad * self._n

            difference = self.calcNorm(wt - wt1)
            wt = wt1
            stepSize += 1

        self._w = wt
        return wt, stepSize

    def calcEout(self):
        Eout = 0

        for i in range(len(self._testSet)):
            X = [1, self._testSet[i][0], self._testSet[i][1]]
            Eout += math.log(1 + e ** (np.dot(-1 * (self._testSet[i][2]), np.dot(self._w, X))))

        return Eout / len(self._testSet)
