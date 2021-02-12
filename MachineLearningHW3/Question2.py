from Data import Data
from LogisticRegression import LogisticRegression

totalStepSize = 0
totalEOut = 0
for i in range(100):
    data = Data(100, 1000)
    w = [0, 0, 0]
    agent = LogisticRegression(.01, data, w)
    w, stepSize = agent.train()
    totalStepSize += stepSize
    totalEOut += agent.calcEout()
print("Average Eout is : ", end="")
print(totalEOut / 100)
print("Average Step Size is : ", end="")
print(totalStepSize / 100)
