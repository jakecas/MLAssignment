from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as npl

import math


def getelsat(x, ilist):
    return [x[i] for i in ilist]


def getcol(x, i):
    return [x[j][i] for j in range(len(x))]


def getcols(x, start, end):
    arr = []
    for i in range(end):
        if i >= start:
            arr.append(getcol(x, i))
    return arr


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



def maketuples(arr):
    return [tuple(i) for i in arr]


def getindexoflargest(arr):
    best = -math.inf
    besti = -1
    for i in range(len(arr)):
        if arr[i] > best:
            best = i
            besti = i
    return besti


def getindexofsmallest(arr):
    best = math.inf
    besti = -1
    for i in range(len(arr)):
        if arr[i] < best:
            best = arr[i]
            besti = i
    return besti


def getgroups(datafile):
    file = open(datafile, "r")
    lines = file.read().split("\n")
    data = []

    for line in lines:
        data.append(line.strip().split(","))

    groups = []
    for i in range(len(data)):
        groups.append(data[i][len(data[i]) - 1])

    nameset = list(set(getcol(data, 4)))
    for g in range(len(groups)):
        for i, name in enumerate(nameset):
            if groups[g] == name:
                groups[g] = i

    return groups


def getdata(datafile, cols, includegroups):
    file = open(datafile, "r")
    lines = file.read().split("\n")
    data = []

    for line in lines:
        data.append(line.strip().split(","))

    dataplot = []
    if len(cols) <= 3:
        for i in range(len(data)):
            datapoint = list(map(float, getelsat(data[i], cols)))
            if includegroups:
                datapoint.append(data[i][len(data[i]) - 1])
            dataplot.append(datapoint)
    else:
        return "ERROR"

    # Determining groups
    if includegroups:
        nameset = list(set(getcol(data, 4)))

        length = len(dataplot[0])
        for el in dataplot:
            for i, name in enumerate(nameset):
                if el[length-1] == name:
                    el[length-1] = i

    return dataplot


def getlabellist(labelfile):
    return open(labelfile).read().split(",")


def inputcols(labelfile):
    print(getlabellist(labelfile))
    numstr = input("Input up to three comma-separated positions of the attributes you wish to plot.\n").split(",")
    return [int(numstr[i]) - 1 for i in range(len(numstr))]


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