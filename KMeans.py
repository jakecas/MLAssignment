import math
import time
import numpy.random as random
import datautils as utils
import copy


def randomcentroids(dataset, k, mindistance):
    centroids = [dataset[i] for i in random.choice(len(dataset), k)]

    for a in centroids:
        for b in centroids:
            if b is not a:
                if utils.euclideandistance(a, b) < mindistance:
                    # if this set of centroids is unfavourable, run it again
                    return randomcentroids(dataset, k, mindistance)
    return centroids


def randomgroups(length, k):
    arr = [i % k for i in range(length)]
    return arr


def findcentroids(dataset, k, groups):
    newcentroids = []
    for i in range(k):
        newcentroids.append([0] * len(dataset[0]))
    total = [0] * k
    for i in range(len(groups)):
        for j in range(len(dataset[i])):
            newcentroids[groups[i]][j] += dataset[i][j]
        total[groups[i]] += 1

    for i in range(len(newcentroids)):
        for p in newcentroids[i]:
            if total[i] is not 0:
                p /= total[i]

    for i in range(k):
        point = []
        for x in newcentroids[i]:
            if total[i] is not 0:
                point.append(x / total[i])
            else:
                point.append(0)
        newcentroids[i] = tuple(point)

    return newcentroids


def kmeans(dataset, k, centroids, groups):
    newgroups = groups.copy()

    for i in range(len(dataset)):
        best = math.inf
        for j in range(k):
            dist = utils.euclideandistance(dataset[i], centroids[j])
            if best > dist:
                best = dist
                newgroups[i] = j

    newcentroids = findcentroids(dataset, k, newgroups)

    if not utils.arraysequal(newcentroids, centroids):
        return kmeans(dataset, k, newcentroids, newgroups)

    return newgroups


def runkmeans(data, k, mindistance):
    return kmeans(data, k, randomcentroids(data, k, mindistance), [0] * len(data))


def plotkmeans(nums, labellist, data_plot, expgroups, k, groupcoulours, centroidsmindist):
    # Plotting
    axessequence = []
    for i in range(len(nums)):
        # Getting transposing the matrix so as to get all x-values in a sequence, y, z.
        axessequence.append(utils.getcol(data_plot, i))

    # Determining groups
    # colours = kmeans(data_plot, k, randomcentroids(data_plot, k), [0] * len(axessequence[0]))

    start = time.time()
    groups = runkmeans(data_plot, k, centroidsmindist)
    end = time.time()
    colours = copy.deepcopy(groups)
    for i, item in enumerate(colours):
        for j, gcolour in enumerate(groupcoulours):
            if item == j:
                colours[i] = gcolour

    utils.nscatter(axessequence, list(labellist[i] for i in nums), colours)
    return accuracy3groups(data_plot, expgroups, groups), end-start


def accuracy3groups(data, expgroups, actgroups):
    tuples = utils.maketuples(data)
    expset = copy.deepcopy(tuples)
    actset = copy.deepcopy(tuples)

    exp = [set(), set(), set()]
    act = [set(), set(), set()]

    for i in range(len(data)):
        exp[expgroups[i]].add(expset[i])
        act[actgroups[i]].add(actset[i])

    bestfit = []
    for e in exp:
        diff = []
        for a in act:
            diff.append(len(e & a))
        bestfit.append(utils.getindexofsmallest(diff))

    if len(set(bestfit)) < 3:
        return 0

    for i in range(len(act)):
        if bestfit[i] == i:
            continue
        else:
            # Switching the current set with the set at the position it should be
            act[bestfit[i]], act[i] = act[i], act[bestfit[i]]
            bestfit[bestfit[i]], bestfit[i] = bestfit[i], bestfit[bestfit[i]]

    finalexp = set()
    finalact = set()
    for i in range(len(exp)):
        for j in exp[i]:
            temp = list(j)
            temp.append(i)
            finalexp.add(tuple(temp))
        for j in act[i]:
            temp = list(j)
            temp.append(i)
            finalact.add(tuple(temp))

    common = finalact & finalexp
    return len(common) / len(finalact)
