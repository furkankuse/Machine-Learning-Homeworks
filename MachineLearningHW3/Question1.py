from math import e


def errorFunc(u, v):  # Error function

    return (u * (e ** v) - 2 * v * (e ** (-u))) ** 2


def deltaU(u, v):  # Derivative of the error function respect to u

    return 2 * (e ** v + 2 * v * (e ** (-u))) * (u * (e ** v) - 2 * v * (e ** (-u)))


def deltaV(u, v):  # Derivative of the error function respect to v

    val = 2 * (u * (e ** v) - 2 * (e ** (-u))) * (u * (e ** v) - 2 * v * (e ** (-u)))
    return val


iter = 0
n = 0.1
w = [1, 1]  # 0th index is u, and 1st index is v

error = 1

while error - 10 ** -14 > 0:  # This one is
    u = deltaU(w[0], w[1])
    v = deltaV(w[0], w[1])
    w[0] -= n * u
    w[1] -= n * v
    error = errorFunc(w[0], w[1])
    iter += 1
print("Number of iterations : ", end="")
print(iter)
print("Weights of u and v : ", end="")
print(w)

w = [1, 1]
for i in range(15):
    w[0] -= deltaU(w[0], w[1]) * n
    w[1] -= deltaV(w[0], w[1]) * n
    error = errorFunc(w[0], w[1])

print("Error of Coordinate Descant is : ", end="")
print(error)
