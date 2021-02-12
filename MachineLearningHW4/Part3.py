from libsvm.svmutil import svm_train
from Data import prepareForOneVsFive, shuffleData, Data


def bestOne(CrossValAccs):  # This function returns the index and the value of given input
    # Used for finding which of the Ecv is minimum
    minVal = 100
    index = 0
    for i in range(len(CrossValAccs)):
        if CrossValAccs[i] < minVal:
            minVal = CrossValAccs[i]
            index = i

    return minVal, index


def problem7and8(trainingData, C):
    trainY, trainX = prepareForOneVsFive(trainingData)
    choices = [0, 0, 0, 0, 0]
    totalError = 0
    for i in range(100):
        trainY, trainX = shuffleData(trainY, trainX)  # libsvm does not randomize input,
        # so before calling train function i call shuffle function from Data and randomize input
        Ecvs = []
        for j in range(len(C)):
            paramters = '-t 1 -d 2 -r 1 -c ' + str(C[j]) + ' -v 10 -q'  # t = 1 indicates kernel will be polynomial,
    # d indicates degree, r indicates + 1 in the polynomial kernel, c is same as C
    # v 10 indicates that 10 fold cross validation will be used
    # and -q prevents it from printing output while training
            Ecvs.append((100 - svm_train(trainY, trainX, paramters)) / 100)

        EcvValue, index = bestOne(Ecvs)
        totalError += EcvValue
        choices[index] += 1

    return (totalError / 100), choices


data = Data('trainData')
trainingData = data.trainingData
C = [0.0001, 0.001, 0.01, 0.1, 1]
Ecv, bestOnes = problem7and8(trainingData, C)
print("-------------------------------------")
print("|     Average Ecv is : {0:f}     |".format(Ecv))
for i in range(len(C)):
    print("|C : {0:5.4f} is chosen for {1:2d} times. |".format(C[i], bestOnes[i]))
print("-------------------------------------")
