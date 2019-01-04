file = open("iris.data", "r")
lines = file.read().split("\n")
data = []

for line in lines:
    data.append(line.split(","))

nums = input(
    "Input up to three comma-separated positions of the attributes you wish to plot (e.g. " + data[0].__str__() + "):\n")
