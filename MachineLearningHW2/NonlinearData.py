import numpy as np


class NonlinearData:

    def __init__(self, trainingSize, testSize, choice):
        self._trainingSet = []  # I init training set
        self._choice = choice
        self.createTrainSet(trainingSize)
        self._testSet = []  # I init test set
        self.createTestSet(testSize)


    def createTrainSet(self, N):
        for i in range(N):
            newPoint = [0, 0]
            newPoint[0], newPoint[1] = np.random.uniform(-1, 1, 2)
            newPoint.append(self.calcYsignEq(newPoint))
            if self._choice:
                if np.random.randint(1, 1000) <= 100:
                    newPoint[2] *= -1
            newPoint.append(0)
            self._trainingSet.append(newPoint)

    def createTestSet(self, N):
        for i in range(N):
            newPoint = [0, 0]
            newPoint[0], newPoint[1] = np.random.uniform(-1, 1, 2)
            newPoint.append(self.calcYsignEq(newPoint))
            self._testSet.append(newPoint)

    def calcYsign(self, line, point):
        value = line[0]
        for i in range(len(line) - 1):
            value += line[i + 1] * point[i]
        return np.sign(value)

    def calcY(self, line, point):
        value = line[0]
        for i in range(len(line) - 1):
            value += line[i + 1] * point[i]
        return value


    def calcYsignEq(self, point):
        return np.sign(point[0] ** 2 + point[1] ** 2 - .6)

    def calcYEq(self, point):
        return point[0] ** 2 + point[1] ** 2 - .6

    @property
    def testSet(self):
        return self._testSet

    @property
    def trainingSet(self):
        return self._trainingSet
