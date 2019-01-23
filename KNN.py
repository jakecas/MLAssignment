import time
import numpy.random as random
import copy

import datautils as utils


def runknn(data, k, percenttraining):
    knninput = copy.deepcopy(data)
    random.shuffle(knninput)
    trainingsize = round(percenttraining * len(knninput))
    trainingset = knninput[:trainingsize].copy()
    unclassified = knninput[trainingsize:].copy()
    for u in unclassified:
        u[len(u)-1] = -1
    knnresult = knn3d(trainingset, unclassified, k)
    return knnresult + trainingset


def knn3d(trainset, dataset, k):
    for x in dataset:
        neighbours = []
        total = [0] * (len(trainset) - 1)
        for p in trainset:
            best = [utils.euclideandistance(x, p)]
            if len(neighbours) < k:
                newn = p + best
                neighbours.append(newn)
            else:
                for n in range(len(neighbours)):
                    last = len(neighbours[n])-1
                    if best[0] < neighbours[n][last]:
                        newn = p + best
                        temp = neighbours[n].copy()[:last]
                        best[0] = neighbours[n][last]
                        neighbours[n], newn = newn, temp

        for n in neighbours:
                total[n[len(n)-2]] += 1

        x[len(x) - 1] = utils.getindexoflargest(total)

    return dataset


def accuracy(expected, actual):
    expectedset = set(utils.maketuples(expected))
    actualset = set(utils.maketuples(actual))
    common = actualset & expectedset

    return len(common) / len(actualset)


def plotknn(nums, labellist, data_plot, k, percenttraining):
    start = time.time()
    fulldata = runknn(data_plot, k, percenttraining)
    end = time.time()

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
    acc = accuracy(data_plot, fulldata)
    # acc = (acc - 100*percenttraining) / ((1-percenttraining) * 100)
    return acc, end-start
