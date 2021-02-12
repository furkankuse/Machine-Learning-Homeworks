from NonlinearData import NonlinearData
from LinearRegresion import LinearRegression
import numpy as np

numberOfTest = 1000
w = [0, 0, 0, 0, 0, 0]
N = 1000
for i in range(numberOfTest):
    data = NonlinearData(N, 1000, False)
    agent = LinearRegression(data, [0, 0, 0, 0, 0], False)
    agent.train()
    w = w + np.dot(agent.w, 1)

print(np.dot(w, 1/numberOfTest))


numberOfTest = 1000
w = [-1, -0.05, .08, .13, 1.5, 1.5]
N = 1000
totalMiss = 0
for i in range(numberOfTest):
    data = NonlinearData(N, 1000, True)
    miss = 0

    for j in range(len(data.trainingSet)):
        point = [data.trainingSet[i][0], data.trainingSet[i][1],data.trainingSet[i][0] * data.trainingSet[i][1], data.trainingSet[i][0] ** 2,data.trainingSet[i][1] ** 2]
        predict = data.calcYsign(w, point)

        if predict != data.trainingSet[i][2]:
            miss += 1

    totalMiss += miss / len(data.trainingSet)

print("Average noisy Eout is: ", end="")
print(totalMiss / numberOfTest)
