from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as npl

import math


# Returns the elements of x at the indexes in ilist
def getelsat(x, ilist):
    return [x[i] for i in ilist]


# Returns the elements at the i'th position of each list in x
def getcol(x, i):
    return [x[j][i] for j in range(len(x))]


# Returns the elements between the start and end position of each list in x
def getcols(x, start, end):
    arr = []
    for i in range(end):
        if i >= start:
            arr.append(getcol(x, i))
    return arr


# Returns true if the lists a and b contain the same elements in the same order
def arraysequal(a, b):
    if len(a) != len(b):
        return False

    for i, j in zip(a, b):
        if i != j:
            return False

    return True


# Calculates and returns the euclidean distance between a and b, using numpy.linalg
def euclideandistance(a, b):
    x = np.array(a)
    y = np.array(b)

    return abs(npl.norm(y-x))


# Returns an array of tuples of the elements in arr
def maketuples(arr):
    return [tuple(i) for i in arr]


# Returns the index of the largest element in arr
def getindexoflargest(arr):
    best = -math.inf  # So that it fails on the first check
    besti = -1
    for i in range(len(arr)):
        if arr[i] > best:
            best = i
            besti = i
    return besti


# Returns the index of the smallest element in arr
def getindexofsmallest(arr):
    best = math.inf  # So that it fails on the first check
    besti = -1
    for i in range(len(arr)):
        if arr[i] < best:
            best = arr[i]
            besti = i
    return besti


# Opens and reads the lines in a comma-separated file and returns the last column,
# except that each element is replaced with a corresponding digit.
def getgroups(datafile):
    file = open(datafile, "r")
    lines = file.read().split("\n")
    data = []

    for line in lines:
        data.append(line.strip().split(","))

    groups = []
    for i in range(len(data)):
        groups.append(data[i][len(data[i]) - 1])

    # Replacing the names with corresponding integers
    nameset = list(set(getcol(data, 4)))
    for g in range(len(groups)):
        for i, name in enumerate(nameset):
            if groups[g] == name:
                groups[g] = i

    return groups


# Obtains the columns 'cols' in the datafile,
# and includes the last column if includegroups is true
def getdata(datafile, cols, includegroups):
    file = open(datafile, "r")
    lines = file.read().split("\n")
    data = []

    for line in lines:
        data.append(line.strip().split(","))

    dataplot = []
    # Since only up to three axes can be plotted, anything more is invalid
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


# Returns a list of the comma-separated elements in labelfile
def getlabellist(labelfile):
    return open(labelfile).read().split(",")


# Prints the elements in labellist and asks for the user to choose up to three
# e.g. 1,4 chooses the 1st and 4th element
def inputcols(labelfile):
    print(getlabellist(labelfile))
    numstr = input("Input up to three comma-separated positions of the attributes you wish to plot.\n").split(",")
    return [int(numstr[i]) - 1 for i in range(len(numstr))]


# Takes up to three lists of values, with each position describing the x, y, and z value of a point respectively,
# it adds lists of zeros if there are less than three,
# then it labels the axes according to labels,
# and plots the data using 'groups' as the colour specifier for each point
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
