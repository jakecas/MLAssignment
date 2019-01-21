from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
import numpy.linalg as npl
import numpy.random as random
import copy

import datautils as utils


def runknn(data, k, percenttraining):
    knninput = copy.deepcopy(data)
    random.shuffle(knninput)
    trainingsize = round(percenttraining * len(knninput))
    trainingset = knninput[:trainingsize].copy()
    unclassified = knninput[trainingsize:].copy()
    knnresult = knn3d(trainingset, unclassified, k)
    return knnresult + trainingset


def knn3d(trainset, dataset, k):
    for x in dataset:
        neighbours = []
        total = [0] * (len(trainset) - 1)
        for p in trainset:
            best = [euclideandistance(x, p)]
            if len(neighbours) < k:
                newn = p + best
                neighbours.append(newn)
            else:
                for n in neighbours:
                    if best[0] < n[len(n)-1]:
                        newn = p + best
                        del n
                        neighbours.append(newn)
                        break

        for n in neighbours:
                total[n[len(n)-2]] += 1

        x[len(x)-1] = utils.getindexoflargest(total)

    return dataset


def accuracy(expected, actual):
    expectedset = set(utils.maketuples(expected))
    actualset = set(utils.maketuples(actual))
    common = actualset & expectedset

    return len(common) / len(actualset)


def arraysequal(a, b):
    if len(a) != len(b):
        return False

    for i, j in zip(a, b):
        if i != j:
            return False

    return True


def euclideandistance(a, b):
    x = np.array(a)
    y = np.array(b)

    return abs(npl.norm(y-x))


def plotknn(nums, labellist, data_plot, k, percenttraining):
    fulldata = runknn(data_plot, k, percenttraining)

    # Converting int class to colours
    colours = utils.getcol(fulldata, len(fulldata[0])-1)
    for i, item in enumerate(colours):
        if item == 0:
            colours[i] = 'b'
        if item == 1:
            colours[i] = 'g'
        if item == 2:
            colours[i] = 'r'

    # Transposing matrix
    axessequence = []
    for i in range(len(fulldata[0])-1):
        axessequence.append(utils.getcol(fulldata, i))

    # print(axessequence)
    utils.nscatter(axessequence, list(labellist[i] for i in nums), colours)
    print("Accuracy: " + str(accuracy(data_plot, fulldata)))
