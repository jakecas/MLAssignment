from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
import numpy.linalg as npl
import numpy.random as random


def randomcentroids(dataset, k):
    return [dataset[i] for i in random.choice(len(dataset), k)]


def findcentroids(dataset, k, groups):
    newcentroids = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    total = [0] * k
    for i in range(len(dataset)):
        newcentroids[groups[i]] = [sum(x) for x in zip(newcentroids[groups[i]], dataset[groups[i]])]
        total[groups[i]] += 1

    for i in range(k):
        newcentroids[i] = tuple(x / total[i] for x in newcentroids[i])

    return newcentroids


def kmeans(dataset, k, centroids, groups):
    newgroups = groups.copy()
    for i in range(len(dataset)):
        best = math.inf
        for j in range(k):
            dist = npl.norm(np.array(tuple(dataset[i])) - np.array(tuple(centroids[j])))
            if best > dist:
                best = dist
                newgroups[i] = j

    for i in range(len(groups)):
        if newgroups[i] != groups[i]:
            print("RECUR!")
            return kmeans(dataset, k, findcentroids(dataset, k, newgroups), newgroups)

    return newgroups


def getelsat(x, ilist):
    return [x[i] for i in ilist]


def getcol(x, i):
    return [x[j][i] for j in range(len(x))]


def nscatter(xyz, labels, groups):
    fig = pyplot.figure()
    ax = Axes3D(fig)

    while len(xyz) < 3:
        xyz.append([0] * len(xyz[0]))
    while len(labels) < 3:
        labels.append("Null")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.scatter(*xyz, c=groups)
    pyplot.show()


file = open("iris.data", "r")
lines = file.read().split("\n")
data = []

for line in lines:
    data.append(line.strip().split(","))

labellist = open("irisattributes.data").read().split(",")
print(labellist)

numstr = input("Input up to three comma-separated positions of the attributes you wish to plot.\n").split(",")
nums = [int(numstr[i])-1 for i in range(len(numstr))]

dataPlot = []
if len(nums) <= 3:
    for i in range(len(data)):
        dataPlot.append(list(map(float, getelsat(data[i], nums))))

# Plotting
axessequence = []
for i in range(len(nums)):
    # Getting transposing the matrix so as to get all x-values in a sequence, y, z.
    axessequence.append(getcol(dataPlot, i))

# Determining groups
kvar = 3
colours = kmeans(dataPlot, kvar, randomcentroids(dataPlot, kvar), [0]*len(axessequence[0]))
for i, item in enumerate(colours):
    if item == 0:
        colours[i] = 'r'
    if item == 1:
        colours[i] = 'g'
    if item == 2:
        colours[i] = 'b'

# print(axessequence)
nscatter(axessequence, list(labellist[i] for i in nums), colours)
