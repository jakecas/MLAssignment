import random
import copy
from KNN import knn3d
from KNN import accuracy
from KNN import plotknn
import datautils as utils
from KMeans import plotkmeans
from KMeans import accuracy3groups
from KMeans import runkmeans


def averageaccuracyknn(dataset, k, percenttraining, runs):
    acc = 0
    for i in range(runs):
        knnin = copy.deepcopy(dataset)
        random.shuffle(knnin)
        trainingslice = round(percenttraining * len(knnin))
        trainset = knnin[:trainingslice].copy()
        unclass= knnin[trainingslice:].copy()
        knnout = knn3d(trainset, unclass, k)
        fullset = knnout + trainset
        acc += accuracy(dataset, fullset)

    return acc


def averageaccuracykmeans(dataset, expgroups, k, centroidsmindist, attempts):
    acc = 0
    actualruns = attempts

    for i in range(attempts):
        actgroups = runkmeans(dataset, k, centroidsmindist)
        out = accuracy3groups(dataset, expgroups, actgroups)
        if out < 0.1:
            acc += out
        else:
            print("Accuracy function failed.")
            actualruns -= 1
    return acc / actualruns


# nums = utils.inputcols("irisattributes.data")
# data_plot = utils.getdata("iris.data", nums, False)
# datagroups = utils.getgroups("iris.data")
# labellist = utils.getlabellist("irisattributes.data")
# # plotknn(nums, labellist, data_plot, 7, 0.45)
# plotkmeans(nums, labellist, data_plot, datagroups, 3, ['r', 'g', 'b'], 1)


# cols = utils.inputcols("irisattributes.data").copy()
# data = utils.getdata("iris.data", cols, True)
# print("Average accuracy over 100 runs: " + str(averageaccuracyknn(data, 17, 0.85, 100)))

cols = utils.inputcols("irisattributes.data").copy()
data = utils.getdata("iris.data", cols, False)
datagroups = utils.getgroups("iris.data")
print("Average accuracy over 100 runs: " + str(averageaccuracykmeans(data, datagroups, 3, 1, 100)))
