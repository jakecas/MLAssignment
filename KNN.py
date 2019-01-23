import time
import numpy.random as random
import copy

import datautils as utils


# Partitions data into a training set and an unclassified set (removing each group to ensure classification),
# and runs the knn3d function, returning the training-set appended to the classified set
def runknn(data, k, percenttraining):
    knninput = copy.deepcopy(data)
    random.shuffle(knninput)
    trainingsize = round(percenttraining * len(knninput))
    trainingset = knninput[:trainingsize].copy()
    unclassified = knninput[trainingsize:].copy()
    for u in unclassified:
        u[len(u)-1] = -1
    knnresult = knn(trainingset, unclassified, k)
    return knnresult + trainingset


# The iterative knn algorithm:
# Finds the closest k points in the training-set to each point in the dataset,
# and classifies it as belonging to the group with the most elements in it's neighborhood
def knn(trainset, dataset, k):
    for x in dataset:
        neighbours = []
        total = [0] * (len(trainset) - 1)
        # Finds the neighboring points
        for p in trainset:
            best = [utils.euclideandistance(x, p)]
            if len(neighbours) < k:
                # Creates the initial neighborhood
                newn = p + best
                neighbours.append(newn)
            else:
                # Checks if this new point should replace any point in the neighborhood
                for n in range(len(neighbours)):
                    last = len(neighbours[n])-1
                    # Replaces the current neighbor,
                    # and continues to check the neighborhood to see
                    # if the replaced neighbor should be placed somewhere else
                    if best[0] < neighbours[n][last]:
                        newn = p + best
                        temp = neighbours[n].copy()[:last]
                        best[0] = neighbours[n][last]
                        neighbours[n], newn = newn, temp

        # Finds the group with the most elements in the neighborhood
        for n in neighbours:
                total[n[len(n)-2]] += 1

        # Classifies the point
        x[len(x) - 1] = utils.getindexoflargest(total)

    return dataset


# Finds the accuracy of the actual set with respect to the
def accuracy(actual, expected):
    actualset = set(utils.maketuples(actual))
    expectedset = set(utils.maketuples(expected))
    common = expectedset & actualset

    return len(common) / len(expectedset)


# Runs Knn and plots the result, returning a tuple of the accuracy and time
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

    utils.nscatter(axessequence, list(labellist[i] for i in nums), colours)
    acc = accuracy(data_plot, fulldata)
    return acc, end-start
