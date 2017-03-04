# Advanced Lane-Finding with OpenCV

While the basic lane-finding project only considers lane lines to be straight lines (a very simplistic approach), here an effort
is made to model and measure lane curvature. Furthermore, more advanced image processing operations are applied to improve
detection of image pixels which may be part of lane lines.

## About/Structure
Important sources are in `/src`. For a description of the project, a writeup describing the steps involved with a number of
supplementary images, see the Jupyter Notebook `advanded_lane_finding.ipynb`. This is the primary entrypoint to the project,
all else is supporting material.

## Developent System
This experiment was run on a MacBook Pro 11'3, with Ubuntu 16.10, running Python 3.5.2 and Anaconda 4.2.0. The
sources depend on OpenCV, numpy, scipy, pandas, and matplotlib.