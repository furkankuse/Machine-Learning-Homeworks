from Data import Data,prepareTrainInput
from libsvm.svmutil import svm_train, svm_predict


def oneVsAll(trainingData):  # This function uses polynomial kernel for svm with soft margin
    # And does same operation for every number in dataset which is 0 to 9
    # returns Eins and number of the support vectors of each number
    EinsofEvens = []
    vectorNumbersOfEven = []
    EinsofOdds = []
    vectorNumbersOfOdd = []
    for i in range(10):
        y, x = prepareTrainInput(i, trainingData)
        model = svm_train(y, x, '-t 1 -d 2 -r 1 -c 0.01 -q')    # t = 1 indicates kernel will be polynomial,
        # d indicates degree, r indicates + 1 in the polynomial kernel, c is same as C
        # and -q prevents it from printing output while training
        support_vectors = model.get_SV()
        p_labels, p_acc, p_vals = svm_predict(y, x, model)
        if i % 2 == 0:
            vectorNumbersOfOdd.append(len(support_vectors))
            EinsofOdds.append((100 - p_acc[0]) / 100)   # acc returns as percent of 100 and it gives 1 - error
            # we do this calculation to make it become error and between 1 and 0
        else:
            vectorNumbersOfEven.append(len(support_vectors))
            EinsofEvens.append((100 - p_acc[0]) / 100)

    return EinsofOdds, vectorNumbersOfOdd, EinsofEvens, vectorNumbersOfEven


data = Data('trainData')
trainingData = data.trainingData

EinsOfOdds, numberOfSupportVectorsOfOdds, EinsOfEvens, numberOfSupportVectorsOfEvens = oneVsAll(trainingData)

print(" -----------------------------------------------")
for i in range(len(EinsOfEvens)):
    print("|{0:d} vs All Ein : {1:f} #Support Vectors = {2:4d}|".format(i * 2, EinsOfEvens[i], numberOfSupportVectorsOfEvens[i]))
print(" -----------------------------------------------")

print(" -----------------------------------------------")
for i in range(len(EinsOfEvens)):
    print("|{0:d} vs All Ein : {1:f} #Support Vectors = {2:4d}|".format(i * 2 + 1, EinsOfOdds[i], numberOfSupportVectorsOfOdds[i]))
print(" -----------------------------------------------")