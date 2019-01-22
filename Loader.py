import datautils as utils


def plotdata(nums, labellist, data_plot):
    # Plotting
    axessequence = []
    for i in range(len(nums)):
        # Transposing the matrix so as to get all x-values in a sequence, y, z.
        axessequence.append(utils.getcol(data_plot, i))

    # Determining groups
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
