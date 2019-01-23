import math
import time
import numpy.random as random
import datautils as utils
import copy


# Returns a list of k random points in dataset which are mindistance apart.
def randomcentroids(dataset, k, mindistance):
    centroids = [dataset[i] for i in random.choice(len(dataset), k)]

    for a in centroids:
        for b in centroids:
            if b is not a:
                if utils.euclideandistance(a, b) < mindistance:
                    # if this set of centroids is unfavourable, run it again
                    return randomcentroids(dataset, k, mindistance)
    return centroids


# Finds the centroids of each group,
# where dataset is a list of points in cartesian space,
# k is the number of groups,
# and groups is the list of groups that each point in dataset belongs to
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


# The recursive kmeans algorithm:
# For each point in dataset,
# it puts it into the group with the closest centroid,
# it finds the centroids of these new groups,
# then it returns the result of the kmeans on these centroids.
# If the groupings do not change, it returns them
def kmeans(dataset, k, centroids, groups):
    newgroups = groups.copy()

    for i in range(len(dataset)):
        best = math.inf
        # Finding the closest centroid
        for j in range(k):
            dist = utils.euclideandistance(dataset[i], centroids[j])
            if best > dist:
                best = dist
                # Setting the group of this point to that of the closest centroid
                newgroups[i] = j

    # Finding the centroids of the new groupings
    newcentroids = findcentroids(dataset, k, newgroups)

    if not utils.arraysequal(newcentroids, centroids):
        # Recursive call with the same dataset, but the new centroids and groupings
        return kmeans(dataset, k, newcentroids, newgroups)

    # Algorithm converged, so no point in running further
    return newgroups


# Starts the kmeans algorithm by finding randomcentroids
def runkmeans(data, k, mindistance):
    return kmeans(data, k, randomcentroids(data, k, mindistance), [0] * len(data))


# Creates a scatter plot of the result of the kmeans algorithm and tries to find the accuracy.
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


# A very unreliable accuracy function, which broke after some changes were made.
# It was not removed due to it *occasionally* working.
def accuracy3groups(data, expgroups, actgroups):
    tuples = utils.maketuples(data)
    expset = copy.deepcopy(tuples)
    actset = copy.deepcopy(tuples)

    exp = [set(), set(), set()]
    act = [set(), set(), set()]

    # Sorting the points into their expected and actual groups.
    for i in range(len(data)):
        exp[expgroups[i]].add(expset[i])
        act[actgroups[i]].add(actset[i])

    # Finding the size of the intersection between each expected and actual group,
    # to try to find the best fit, since there is no guarantee the classes are called the same.
    bestfit = []
    for e in exp:
        diff = []
        for a in act:
            diff.append(len(e & a))
        bestfit.append(utils.getindexofsmallest(diff))

    if len(set(bestfit)) < 3:
        # No best fit found
        return 0

    # Shifting the sets to their proper place
    for i in range(len(act)):
        if bestfit[i] == i:
            continue
        else:
            # Switching the current set with the set at the position it should be
            act[bestfit[i]], act[i] = act[i], act[bestfit[i]]
            bestfit[bestfit[i]], bestfit[i] = bestfit[i], bestfit[bestfit[i]]

    # Adding the points to a single set with their class appended
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

    # Finding the common points across the actual and expected sets, and thus the accuracy.
    common = finalact & finalexp
    return len(common) / len(finalact)
