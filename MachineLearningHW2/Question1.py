from math import log


def calculateN(M, epsilon, probability):

    return log(probability / (2 * M)) / (-2 * (epsilon ** 2))


print(calculateN(1, 0.05, 0.03))
print(calculateN(10, 0.05, 0.03))
print(calculateN(100, 0.05, 0.03))
