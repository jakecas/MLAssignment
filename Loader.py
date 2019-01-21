import datautils as utils


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
        dataPlot.append(list(map(float, utils.getelsat(data[i], nums))))

# Plotting
# axesmax = []
axessequence = []
for i in range(len(nums)):
    # Finding maximum of each column.
    # axesmax.append(max(dataPlot, key=lambda x: x[i])[i])
    # Getting transposing the matrix so as to get all x-values in a sequence, y, z.
    axessequence.append(utils.getcol(dataPlot, i))

# Determining groups
colours = utils.getcol(data, 4)
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
utils.nscatter(axessequence, list(labellist[i] for i in nums), colours)
