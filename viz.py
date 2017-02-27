# The viz module contains functions for drawing helpful graphics at
# various stages of the pre-processing pipeline.
import matplotlib.pyplot as plt
import numpy as np

def mosaic(images, height, width, ncol, cmap=None):
    """ Produce a mosaic from the input images.

    Images are plotted one next to the other, in ncol columns.
    They will take up the space defined by the width/height
    parameters. The number of rows is determine automatically.

    Args:
    images: Images to be plotted
    height: Plotting area height
    width: Plotting area width
    ncol: Number of columns in mosaic

    Returns:
    A pyplot figure.
    """

    figure = plt.figure(figsize=(width, height))


    for i in range(0, len(images)):
        figure_ax = figure.add_subplot(np.ceil(len(images)/ncol), ncol, i+1)

        figure_ax.imshow(images[i], figure=figure, aspect='auto', interpolation='nearest', cmap=cmap)
        figure_ax.axis('off')

    figure.subplots_adjust(wspace=0, hspace=0)

    return figure
