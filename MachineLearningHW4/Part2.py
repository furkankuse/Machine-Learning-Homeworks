from Data import Data, prepareForOneVsFive
from libsvm.svmutil import svm_train, svm_predict


def oneVsFive(C, Q, y, x, yTest, xTest): # This function uses polynomial kernel for svm with soft margin
    # And does this operation for only ones and fives from dataset
    # it return Eout, Ein and number of the support vectors
    parameters = '-t 1 -d ' + str(Q) + ' -r 1 -c ' + str(C) + ' -q'  # t = 1 indicates kernel will be polynomial,
    # d indicates degree, r indicates + 1 in the polynomial kernel, c is same as C
    # and -q prevents it from printing output while training
    model = svm_train(y, x, parameters)
    support_vectors = model.get_SV()
    p_labels, p_acc, p_vals = svm_predict(yTest, xTest, model)
    Eout = (100 - p_acc[0]) / 100  # acc returns as percent of 100 and it gives 1 - error
    # we do this calculation to make it become error and between 1 and 0
    p_labels, p_acc, p_vals = svm_predict(y, x, model)
    Ein = (100 - p_acc[0]) / 100
    numberOfSupportVectors = len(support_vectors)

    return Eout, Ein, numberOfSupportVectors


def problem5and6(Q, C):
    trainY, trainX = prepareForOneVsFive(trainingData)
    testY, testX = prepareForOneVsFive(testData)
    Eouts = []
    Eins = []
    numbersOfVectors = []
    for i in range(len(C)):
        Eout, Ein, numberOfVectors = oneVsFive(C[i], Q, trainY, trainX, testY, testX)
        Eouts.append(Eout)
        Eins.append(Ein)
        numbersOfVectors.append(numberOfVectors)

    return Eouts, Eins, numbersOfVectors


data = Data('trainData', 'testData')
trainingData = data.trainingData
testData = data.testData
C = [0.0001, 0.001, 0.01, 0.1, 1]

Eouts1, Eins1, numbersOfVectors1 = problem5and6(2, C)
Eouts2, Eins2, numbersOfVectors2 = problem5and6(5, C)

print(" ---------------------------For Q = 2----------------------------")
for i in range(len(C)):
    print("|C = {0:5.4f} Ein = {1:f} Eout = {2:f} #Support Vectors = {3:3d}|".format(C[i], Eins1[i], Eouts1[i],
                                                                                     numbersOfVectors1[i]))
print(" ----------------------------------------------------------------")
print(" ---------------------------For Q = 5----------------------------")
for i in range(len(C)):
    print("|C = {0:5.4f} Ein = {1:f} Eout = {2:f} #Support Vectors = {3:3d}|".format(C[i], Eins2[i], Eouts2[i],
                                                                                     numbersOfVectors2[i]))
print(" ----------------------------------------------------------------")
