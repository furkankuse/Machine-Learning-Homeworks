from Data import Data
from PLA import PLA

totalStep = 0
totalMissChange = 0
numberOfTest = 1000
N = 100
for i in range(numberOfTest):
    data = Data(N, 1000)
    agent = PLA(data, [0, 0, 0])
    step, w = agent.train()
    print("Episode: ", end="")
    print(i + 1)
    print("Step size: ", end="")
    print(step)
    missChange = agent.calcProb()
    totalMissChange += missChange
    totalStep += step
print("Average step size is: ", end="")
print(totalStep / numberOfTest)
print("Average miss change is: ", end="")
print(totalMissChange / numberOfTest)