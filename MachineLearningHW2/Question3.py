from Data import Data
from LinearRegresion import LinearRegression
from PLA import PLA

numberOfTest = 1000
totalEin = 0
totalEout = 0
N = 100
for i in range(numberOfTest):
    data = Data(N, 1000)
    agent = LinearRegression(data, [0, 0, 0], True)
    Ein = agent.train() / N
    totalEin += Ein
    totalEout += agent.calcEout()
print("Average Ein is : ", end="")
print(totalEin / numberOfTest)
print("Average Eout is : ", end="")
print(totalEout / numberOfTest)

numberOfTest1 = 1000
totalStep1 = 0
N1 = 10
for i in range(numberOfTest1):
    data = Data(N1, 1000)
    agent = LinearRegression(data, [0, 0, 0], True)
    agent.train()
    w = agent.w
    agentPla = PLA(data, w)
    step, w = agentPla.train()
    print("Episode: ", end="")
    print(i + 1)
    print("Step size: ", end="")
    print(step)
    totalStep1 += step

print("Average step size is: ", end="")
print(totalStep1 / numberOfTest1)