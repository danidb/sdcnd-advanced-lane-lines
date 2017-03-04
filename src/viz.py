# The viz module contains functions for drawing helpful graphics at
# various stages of the pre-processing pipeline.
import matplotlib.pyplot as plt
import numpy as np
import cv2
from util import eval_poly

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


def lane_overlay(binary_image, left_fit, right_fit):
    """ Get the lane overlay, showing the detected lane
x
    Args:
    binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
    should have been applied to the image prior to this step. The image must be binary, and
    contain only values in {0, 1}.

    left_fit: Polynomial fit to the left lane line
    right_fit: Polynomial fit to the right lane line

    Returns:
    Green overlay, do be drawn on a calibrated color image, showing detected lane line
    between left_fit and right_fit.
    """

    ## The lane is represented as an overlay between the two lane lines, which
    ## are taken to be the polynomials defined by left_fit and right_fit.
    plot_y = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0])

    left_fit_x = eval_poly(plot_y, left_fit)
    left_points = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])

    right_fit_x = eval_poly(plot_y, right_fit)
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])

    points = np.hstack((left_points, right_points))

    return_image = np.zeros((1280, 720, 3), dtype=np.uint8)

    cv2.fillPoly(return_image, np.int_([points]), (0, 255, 0))

    return return_image
