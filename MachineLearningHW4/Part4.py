from libsvm.svmutil import svm_train, svm_predict
from Data import prepareForOneVsFive, Data


def problem9and10(trainingData, testData, C):
    trainY, trainX = prepareForOneVsFive(trainingData)
    testY, testX = prepareForOneVsFive(testData)
    Eins = []
    Eouts = []
    for i in range(len(C)):
        parameters = '-t 2 -c ' + str(C[i]) + ' -q'  # t = 2 indicates kernel will be RBF,
    # c is same as C and -q prevents it from printing output while training
        model = svm_train(trainY, trainX, parameters)
        p_labels, p_acc, p_vals = svm_predict(testY, testX, model)
        Eouts.append((100 - p_acc[0]) / 100)  # acc returns as percent of 100 and it gives 1 - error
    # we do this calculation to make it become error and between 1 and 0
        p_labels, p_acc, p_vals = svm_predict(trainY, trainX, model)
        Eins.append((100 - p_acc[0]) / 100)

    return Eins, Eouts


data = Data('trainData', 'testData')
trainingData = data.trainingData
testData = data.testData
C = [0.01, 1, 100, 10000, 1000000]
Eins, Eouts = problem9and10(trainingData, testData, C)

print("-----------------------------------------------")
for i in range(len(C)):
    print("|C = {0:10.2f} Ein = {1:f} Eout = {2:f}|".format(C[i], Eins[i], Eouts[i]))

print("-----------------------------------------------")
