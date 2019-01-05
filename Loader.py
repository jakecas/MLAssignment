from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


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
# axesmax = []
axessequence = []
for i in range(len(nums)):
    # Finding maximum of each column.
    # axesmax.append(max(dataPlot, key=lambda x: x[i])[i])
    # Getting transposing the matrix so as to get all x-values in a sequence, y, z.
    axessequence.append(getcol(dataPlot, i))

# Determining groups
colours = getcol(data, 4)
nameset = list(set(colours))
for i, el in enumerate(colours):
    if el == nameset[0]:
        colours[i] = 'b'
    if el == nameset[1]:
        colours[i] = 'g'
    if el == nameset[2]:
        colours[i] = 'r'

# print(axesmax)
print(axessequence)
nscatter(axessequence, list(labellist[i] for i in nums), colours)
