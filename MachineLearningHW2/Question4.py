from NonlinearData import NonlinearData
from LinearRegresion import LinearRegression

numberOfTest = 1000
totalEin = 0
w = []
N = 1000
for i in range(numberOfTest):
    data = NonlinearData(N, 1000, True)
    agent = LinearRegression(data, [0, 0, 0], True)
    Ein = agent.train() / N
    w = agent.w
    totalEin += Ein

print("Average Ein : ", end="")
print(totalEin / numberOfTest)
