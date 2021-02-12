import numpy as np


class Data:

    def __init__(self, trainingSize, testSize):
        self._point1 = np.random.uniform(-1, 1, 2)  # I create 2 points for creating target line
        self._point2 = np.random.uniform(-1, 1, 2)

        self._slope = (self.point1[1] - self.point2[1]) / (self.point1[0] - self.point2[0])
        self._X0 = self.point1[1] - self._slope * self.point1[0]  # I calculate coefficients of the line

        self._trainingSet = []  # I init training set
        self.createTrainSet(trainingSize)
        self._testSet = []  # I init test set
        self.createTestSet(testSize)

    def createTrainSet(self, N):
        for i in range(N):  # Filling the set with random points
            newPoint = [0, 0]
            newPoint[0], newPoint[1] = np.random.uniform(-1, 1, 2)
            newPoint.append(self.calcYsign([self._X0, self._slope, -1], newPoint))  # give y values from target function
            self._trainingSet.append(newPoint)

    def createTestSet(self, N):
        for i in range(N):  # Filling the test set with random poits
            newPoint = [0, 0]
            newPoint[0], newPoint[1] = np.random.uniform(-1, 1, 2)
            newPoint.append(self.calcYsign([self._X0, self._slope, -1], newPoint))  # give y values from target function
            self._testSet.append(newPoint)

    @property
    def testSet(self):
        return self._testSet

    @property
    def line(self):
        return [self._X0, self._slope, -1]

    @property
    def trainingSet(self):
        return self._trainingSet

    @property
    def point1(self):
        return self._point1

    @property
    def point2(self):
        return self._point2

    def calcYsign(self, line, point):
        return np.sign(line[0] + line[1] * point[0] + line[2] * point[1])

    def calcY(self, line, point):
        return line[0] + line[1] * point[0] + line[2] * point[1]
