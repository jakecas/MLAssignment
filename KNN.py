from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
import numpy.linalg as npl
import numpy.random as random


def discreteknn(dataset, k):
    random.shuffle(dataset)
    trainingsize = round(0.25 * len(dataset))
    trainingset = dataset[:trainingsize]
    unclassified = dataset[trainingsize:]
    neighbours = []
    total = [0] * (len(dataset[0])-1)

    for p in unclassified:
        for q in trainingset:
            best = euclideandistance(q[:len(dataset)-1], p[:len(dataset)-1])
            if len(neighbours) == k:
                for i in neighbours:
                    if best < i[len(i)-1]:
                        i = q.append(best)
                        break
            else:
                neighbours.append(q.append(best))
        for n in neighbours:
            total[n[len(n)-2]] += 1
        p[len(p)-2] = getindexoflargest(total)

    return trainingset.append(unclassified[:len(unclassified)-1])


def getindexoflargest(arr):
    best = -math.inf
    for i in arr:
        if i > best:
            best = i
    return best


def getelsat(x, ilist):
    return [x[i] for i in ilist]


def getcol(x, i):
    return [x[j][i] for j in range(len(x))]


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
        datapoint = list(map(float, getelsat(data[i], nums)))
        datapoint.append(data[i][len(data[i])-1])
        dataPlot.append(datapoint)

# Determining groups
nameset = list(set(getcol(data, 4)))

length = len(dataPlot[0])
for i, el in enumerate(dataPlot):
    if el[length-1] == nameset[0]:
        el[length-1] = 0
    if el == nameset[1]:
        el[length-1] = 1
    if el == nameset[2]:
        el[length-1] = 2

kvar = 7
knnresult = discreteknn(dataPlot, kvar)

# Transposing matrix
axessequence = []
colours = []
for i in range(len(knnresult)):
    if i < len(knnresult)-1:
        axessequence.append(getcol(knnresult, i))
    else:
        colours = getcol(knnresult, i)

# print(axessequence)
nscatter(axessequence, colours, knnresult[len(knnresult)-1])
