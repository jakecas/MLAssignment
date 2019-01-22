import random
import copy
import time
from Loader import plotdata
from KNN import knn3d
from KNN import accuracy
from KNN import plotknn
import datautils as utils
from KMeans import plotkmeans
from KMeans import accuracy3groups
from KMeans import runkmeans


def averageaccuracyknn(dataset, k, percenttraining, runs):
    acc = 0
    time100 = 0
    for i in range(runs):
        knnin = copy.deepcopy(dataset)
        random.shuffle(knnin)
        trainingslice = round(percenttraining * len(knnin))
        trainset = knnin[:trainingslice].copy()
        unclass = knnin[trainingslice:].copy()

        start = time.time()
        knnout = knn3d(trainset, unclass, k)
        end = time.time()

        fullset = knnout + trainset
        acc += accuracy(dataset, fullset)
        time100 += end - start

    acc = (acc - 100*percenttraining) / ((1-percenttraining) * 100)
    return acc, time100


def averageaccuracykmeans(dataset, expgroups, k, centroidsmindist, attempts):
    acc = 0
    actualruns = attempts
    time100 = 0

    for i in range(attempts):
        start = time.time()
        actgroups = runkmeans(dataset, k, centroidsmindist)
        end = time.time()

        out = accuracy3groups(dataset, expgroups, actgroups)
        if out > 0.1:
            acc += out
            time100 += end - start
        else:
            actualruns -= 1
    if actualruns == 0:
        return "Accuracy function failed."
    return actualruns, acc / actualruns, time100


choice = -1
while choice != 6:
    print("=========Menu=========")
    print("1. Plot Dataset")
    print("2. K-Means")
    print("3. Average K-Means")
    print("4. KNN")
    print("5. Average KNN")
    print("6. Quit")
    print("======================")
    choice = int(input("Enter the number of the desired option: "))

    cols = utils.inputcols("irisattributes.data").copy()
    labellist = utils.getlabellist("irisattributes.data")

    if choice == 1:
        data = utils.getdata("iris.data", cols, True)
        plotdata(cols, labellist, data)
    elif choice == 2:
        data = utils.getdata("iris.data", cols, False)
        datagroups = utils.getgroups("iris.data")

        print("Accuracy and time: " + str(plotkmeans(cols, labellist, data, datagroups, 3, ['r', 'g', 'b'], 1)))
    elif choice == 3:
        data = utils.getdata("iris.data", cols, False)
        datagroups = utils.getgroups("iris.data")
        print("Runs, average accuracy, time: " + str(averageaccuracykmeans(data, datagroups, 3, 1, 100)))

    elif choice == 4:
        data = utils.getdata("iris.data", cols, True)
        kvar = int(input("Enter k: "))
        perc = float(input("Enter percent training set: ")) / 100
        print("Accuracy and time: " + str(plotknn(cols, labellist, data, kvar, perc)))
    elif choice == 5:
        data = utils.getdata("iris.data", cols, True)
        kvar = int(input("Enter k: "))
        perc = float(input("Enter percent training set: "))/100
        print("Accuracy and time over 100 runs: " + str(averageaccuracyknn(data, kvar, perc, 100)))
