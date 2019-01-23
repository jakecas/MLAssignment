import datautils as utils


# Plots the points in data_plot with colours according to their class (from the last element),
# and labels each axis appropriately
def plotdata(nums, labellist, data_plot):
    # Plotting
    axessequence = []
    for i in range(len(nums)):
        # Transposing the matrix so as to get all x-values in a sequence, similarly for y, z.
        axessequence.append(utils.getcol(data_plot, i))

    # Determining colours according to groups
    colours = utils.getcol(data_plot, len(data_plot[0])-1)
    nameset = list(set(colours))
    for i, el in enumerate(colours):
        if el == nameset[0]:
            colours[i] = 'b'
        if el == nameset[1]:
            colours[i] = 'g'
        if el == nameset[2]:
            colours[i] = 'r'

    utils.nscatter(axessequence, list(labellist[i] for i in nums), colours)
