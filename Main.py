import random
import copy
import time
from Loader import plotdata
from KNN import runknn
from KNN import accuracy
from KNN import plotknn
import datautils as utils
from KMeans import plotkmeans
from KMeans import accuracy3groups
from KMeans import runkmeans


# Runs KNN runs times on the dataset with the appropriate k and percentage of the dataset used.
# Returns a tuple of the accuracy (%) and time taken.
def averageaccuracyknn(dataset, k, percenttraining, runs):
    acc = 0
    time100 = 0
    for run in range(runs):
        start = time.time()
        knnout = runknn(dataset, k, percenttraining)
        end = time.time()

        temp = accuracy(dataset, knnout)
        acc += temp
        time100 += end - start

    # acc = (acc - 100.0*percenttraining) / ((1.0-percenttraining) * 100.0)
    return acc, time100


# Runs K-Means 100 times and tries to calculate the accuracy.
# Returns a tuple of the runs with accuracy completed, the accuracy (as a fraction), and the time taken.
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
labellist = utils.getlabellist("irisattributes.data")

# The menu
while choice != 8:
    data = []
    print("===================Menu===================")
    print("1. Plot Dataset")
    print("2. K-Means")
    print("3. Average K-Means")
    print("4. KNN")
    print("5. Average KNN")
    print("6. Tabulate KNN over size of training set")
    print("7. Tabulate KNN over k")
    print("8. Quit")
    print("==========================================")
    choice = int(input("Enter the number of the desired option: "))
    if choice == 8:
        break

    cols = utils.inputcols("irisattributes.data").copy()

    if choice == 1:
        # Task 1; Data Visualization
        data = utils.getdata("iris.data", cols, True)
        plotdata(cols, labellist, data)

    elif choice == 2:
        # Task 2; K-Means
        data = utils.getdata("iris.data", cols, False)
        datagroups = utils.getgroups("iris.data")
        print("Accuracy and time: " + str(plotkmeans(cols, labellist, data, datagroups, 3, ['r', 'g', 'b'], 1)))

    elif choice == 3:
        # Average accuracy of K-Means
        data = utils.getdata("iris.data", cols, False)
        datagroups = utils.getgroups("iris.data")
        print("Runs, average accuracy, time: " + str(averageaccuracykmeans(data, datagroups, 3, 1, 100)))

    elif choice == 4:
        # Task 3; KNN
        data = utils.getdata("iris.data", cols, True)
        kvar = int(input("Enter k: "))
        perc = float(input("Enter percent training set: ")) / 100
        print("Accuracy (including training set) and time: " + str(plotknn(cols, labellist, data, kvar, perc)))

    elif choice == 5:
        # Average accuracy of KNN
        data = utils.getdata("iris.data", cols, True)
        kvar = int(input("Enter k: "))
        perc = float(input("Enter percent training set: "))/100.0
        print("Accuracy and time over 100 runs: " + str(averageaccuracyknn(data, kvar, perc, 100)))

    elif choice == 6:
        # Tabulates KNN with a constant K, varying size of training set between 15% and 85%
        data = utils.getdata("iris.data", cols, True)
        kvar = int(input("Enter k: "))
        for i in range(8):
            perc = (15 + i*10) / 100.0
            print(str(averageaccuracyknn(data, kvar, perc, 100)))

    elif choice == 7:
        # Tabulates KNN with a constant size of training set, varying K as n+1 where n are the multiples of three
        data = utils.getdata("iris.data", cols, True)
        perc = float(input("Enter percent training set: "))/100.0
        for i in range(round(150*perc/3)):
            kvar = 1 + i*3
            print(str(kvar) + " " + str(averageaccuracyknn(data, kvar, perc, 100)))
