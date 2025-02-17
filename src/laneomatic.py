import numpy as np
from math import log2
import cv2
from util import eval_poly, exp_smooth

class LaneOMatic:
    """ The 'Lane-O-Matic' class defines a lane line detection procedure.

    The procedure defined within this class, relies on pre-processed input, and
    the original source image. This class contains facilities for detecting lane
    lines via a sliding window.

    For videos, the lane line detection procedure can be smoothed by taking into
    account the estimates for the last N frames. Exponential smoothing is applied.

    Args:
    primary_only: Should we only apply primary detection? If LaneOMatic is being
    applied to a series of un-related still images, set this to True and the same
    instance can be used for all detection, avoiding repitition.

    smoother_gamma: Decay parameter (exponential, should be in [0,1]) for smoothing.

    n_windows: Number of sliding windows.

    window_width: Width of windows. Consider a mass of lane line pixels to be the centre,
    and the window will extend +/- window_width/2 about that centre.

    window_threshold:
    """

    def __init__(self, primary_only=False, smoother_gamma=0.2, n_windows=10, window_width=100, window_threshold=50):
        self.primary_only = primary_only
        self.smoother_gamma = smoother_gamma
        self.n_windows = n_windows
        self.window_width = window_width
        self.window_threshold = window_threshold
        self.previous_fit = None


    def detect_lanes(self, binary_image, image_only=False) :
        """ Detect the lane lines in the provided binary image, and return the polynomial fit.

        Args:
        binappry_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        image_only: Return image only? Or include curvature. Default is False (return everything)

        Returns:
        Output image, with detection information superimposed.
        """
        if self.previous_fit != None:
            curvatures, fits, deviation, output_image = self.secondary_detection(binary_image)
        else:
            curvatures, fits, deviation, output_image = self.primary_detection(binary_image)

        if not self.primary_only:
            self.previous_fit = fits

        left_fit, right_fit = fits[0], fits[1]
        left_curvature, right_curvature = curvatures[0], curvatures[1]

        if image_only:
            return output_image
        else:
            return left_fit, left_curvature, right_fit, right_curvature, deviation, output_image

    def secondary_detection(self, binary_image):
        """ Detection for frames of video after the first.

        The sliding window technique applied for primary lane detection is expensive.
        Given that a model already exists for the left and right lane lines, it is
        reasonable to assume that the curves defined by these points are a suitable
        starting point when lanes are to be detected on the next frame. Here, the parameters
        of the curves are updated based on the points that surround them. To guard against
        problems with individual frames, exponential smoothing is applied to the parameters
        of the fit.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        Returns:
        ((left_fit, right_fit), output_image). The fit parameters have been smoothed.
        """
        image_1x, image_1y = self._image_nonzero(binary_image)

        prev_left_fit, prev_right_fit = self.previous_fit

        ## Pixels in the region surrounding each previous lane line are detected, and
        ## used to update the parameters of the curve.
        left_lane = ((image_1x > eval_poly(image_1y, prev_left_fit) - self.window_width/2) &
                     (image_1x < eval_poly(image_1y, prev_left_fit) + self.window_width/2))
        right_lane = ((image_1x > eval_poly(image_1y, prev_right_fit) - self.window_width/2) &
                      (image_1x < eval_poly(image_1y, prev_right_fit) + self.window_width/2))

        if len(left_lane) > self.window_threshold:
            left_fit = self._lane_fit(left_lane, image_1x, image_1y)
        else:
            left_fit = self.previous_fit[0]

        if len(right_lane) > self.window_threshold:
            right_fit = self._lane_fit(right_lane, image_1x, image_1y)
        else:
            right_fit = self.previous_fit[1]

        ## Apply exponential smoothing to the parameters of the fit.
        left_fit = [exp_smooth(x, s_p, self.smoother_gamma) for x, s_p in zip(left_fit, self.previous_fit[0])]
        right_fit = [exp_smooth(x, s_p, self.smoother_gamma) for x, s_p in zip(right_fit, self.previous_fit[1])]

        output_image = self._draw_secondary_detection(binary_image, left_lane, left_fit,
                                                      right_lane, right_fit,
                                                      image_1x, image_1y)

        ## Compute curvature, and deviation from centre, and write log_2(curvature) (right, left) onto the image
        left_centre, left_curvature = self._compute_curvature(binary_image, left_lane, image_1x, image_1y)
        right_centre, right_curvature = self._compute_curvature(binary_image, right_lane, image_1x, image_1y)

        curvature_text = "log2(curvature): (L: "+str(round(log2(left_curvature), 3))+", R: "+str(round(log2(right_curvature), 3))+")"
        cv2.putText(output_image, curvature_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        deviation = (binary_image.shape[1]/2 * 3.7/700) - np.mean((left_centre, right_centre))
        deviation_text = "offset: " + str(round(deviation, 2))
        cv2.putText(output_image, deviation_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return ((left_curvature, right_curvature), (left_fit, right_fit), deviation, output_image)

    def _draw_secondary_detection(self, binary_image, left_lane, left_fit, right_lane, right_fit,
                                  image_1x, image_1y):
        """ Draw the results of secondary detection, frames of video other than the first.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        left_lane: Pixels determined to be part of the left lane
        left_fit : Polynomial fit to the left lane
        right_lane: Pixels determined to be part of the right lane
        right_fit: Polynomial fit to the right lane

        image_1x: Nonzero pixels (x-coord)
        image1y: Nonzero pixels (y-coord)

        Returns:
        Image with lane lines highlighted, and fit illustrated.
        """

        ## See _draw_primary_detection for further details.

        output_image = np.dstack((binary_image, binary_image, binary_image))*255
        output_image[image_1y[left_lane], image_1x[left_lane]] = [255, 0, 0]
        output_image[image_1y[right_lane], image_1x[right_lane]] = [0, 0, 255]

        window_image = np.zeros_like(output_image)

        plot_y = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0], dtype=np.int32)

        left_fit_x = np.array(eval_poly(plot_y, left_fit), dtype=np.int32)
        left_lane_left = np.array([np.transpose(np.vstack([left_fit_x - np.int(self.window_width/2), plot_y]))])
        left_lane_right = np.array([np.transpose(np.vstack([left_fit_x + np.int(self.window_width/2), plot_y]))])

        left_lane_points = np.hstack((left_lane_left, left_lane_right))

        for left_fit_point in zip(left_fit_x, plot_y):
            cv2.circle(output_image, left_fit_point, 1, (255, 255, 0))

        right_fit_x = np.array(eval_poly(plot_y, right_fit), dtype=np.int32)
        right_lane_left = np.array([np.transpose(np.vstack([right_fit_x - np.int(self.window_width/2), plot_y]))])
        right_lane_right = np.array([np.transpose(np.vstack([right_fit_x + np.int(self.window_width/2), plot_y]))])

        right_lane_points = np.hstack((right_lane_left, right_lane_right))

        for right_fit_point in zip(right_fit_x, plot_y):
            cv2.circle(output_image, right_fit_point, 1, (255, 255, 0))

        cv2.addWeighted(output_image, 1, window_image, 0.3, 0)

        return output_image

    def primary_detection(self, binary_image):
        """ Detect the lane lines in an image, or the first frame of a video.

        The lane line procedure is based on a sliding window. The initial
        phase of the procedure uses a histogram to select two starting points
        for the columns of windows. This method is the first step to detecting
        lane lines in a video, or the only step required to detect lines in a
        single image.

p        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        Returns:
        ((left_fit, right_fit), output_image)
        """

        image_1x, image_1y = self._image_nonzero(binary_image)

        ## The starting points are determined by from a histogram of the
        ## bottom half of the image. They correspond to the position of the
        ## left and right peaks of ths histogram, which we assume represent
        ## lane lines.
        left_x_init, right_x_init = self._lane_init(binary_image)
        left_x, right_x = left_x_init, right_x_init

        left_lanes, right_lanes = [], []
        left_windows, right_windows = [], []

        ## Proceeding vertically, from top to bottom, we collect data from window, first the left
        ## and then the right. Centroids are updated accordingly prior to the next round.
        ## A number of supporting methods are used here to make this read more like a template,
        ## this structure should be adhered to elsewhere.
        for wi in range(self.n_windows):

            left_window = self._window_boundaries(wi, left_x, binary_image.shape)
            left_windows.append(left_window)

            left_lane = self._init_lane_pixels(left_window['x0'], left_window['y0'],
                                               left_window['x1'], left_window['y1'],
                                               image_1x, image_1y)
            left_lanes.append(left_lane)


            right_window = self._window_boundaries(wi, right_x, binary_image.shape)
            right_windows.append(right_window)
            right_lane = self._init_lane_pixels(right_window['x0'], right_window['y0'],
                                                right_window['x1'], right_window['y1'],
                                                image_1x, image_1y)
            right_lanes.append(right_lane)

            if len(left_lane) > self.window_threshold:
                left_x = np.int(np.mean(image_1x[left_lane]))
            if len(right_lane) > self.window_threshold:
                right_x = np.int(np.mean(image_1x[right_lane]))


        ## _lane_fit will fit a quadratic polynomial to the line.
        left_fit  = self._lane_fit(np.concatenate(left_lanes), image_1x, image_1y)
        right_fit = self._lane_fit(np.concatenate(right_lanes), image_1x, image_1y)

        output_image = self._draw_primary_detection(binary_image, left_windows, np.concatenate(left_lanes), left_fit,
                                                    right_windows, np.concatenate(right_lanes), right_fit, image_1x, image_1y)

        ## Compute curvature, and deviation from centre, and write log_2(curvature) (right, left) onto the image
        left_centre, left_curvature = self._compute_curvature(binary_image, np.concatenate(left_lanes), image_1x, image_1y)
        right_centre, right_curvature = self._compute_curvature(binary_image, np.concatenate(right_lanes), image_1x, image_1y)

        curvature_text = "log2(curvature): (L: "+str(round(log2(left_curvature), 3))+", R: "+str(round(log2(right_curvature), 3))+")"
        cv2.putText(output_image, curvature_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        deviation = (binary_image.shape[1]/2 * 3.7/700) - np.mean((left_centre, right_centre))
        deviation_text = "offset: " + str(round(deviation, 2))
        cv2.putText(output_image, deviation_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        return ((left_curvature, right_curvature), (left_fit, right_fit), deviation, output_image)


    def _draw_primary_detection(self, binary_image, left_windows, left_pixels,  left_fit,
                                right_windows, right_pixels, right_fit, image_1x, image_1y):
        """ Draw sliding windows on an image, with detected lane lines.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        left_windows: List of coordinates. The windows used to detect the left lane.
        left_pixels: Numpy array. Pixels detected to be part of the left lane.
        left_fit: Numpy fit. Polynomial fit to the left lane.

        right_windows: List of coordinates. The windows used to detect the right lane.
        right_pixels: Numpy array. Pixels detected to be part of the right lane.
        right_fit: Numpy fit. Polynomial fit to the right lane.


        Returns:
        Image ready for display, with windows, detected lane-line points, and
        the polynomial fit to the lanes.
        """

        output_image = np.dstack((binary_image, binary_image, binary_image))*255

        ## This gives one point for each y position in the image, used later to illustrate the
        ## lane line models.
        plot_y = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0], dtype=np.int32)

        left_fit_x = np.array(eval_poly(plot_y, left_fit), dtype=np.int32)
        right_fit_x = np.array(eval_poly(plot_y, right_fit), dtype=np.int32)

        ## The lane line pixels are filled with red, and blue to easily
        ## differentiate between the two lanes. Windows are outlined in green.
        output_image[image_1y[left_pixels], image_1x[left_pixels]] = [255, 0, 0]
        output_image[image_1y[right_pixels], image_1x[right_pixels]] = [0, 0, 255]

        for window in left_windows:
            cv2.rectangle(output_image,
                          (window['x0'], window['y0']),
                          (window['x1'], window['y1']),
                          (0, 255, 0), 2)

        for window in right_windows:
            cv2.rectangle(output_image,
                          (window['x0'], window['y0']),
                          (window['x1'], window['y1']),
                          (0, 255, 0), 2)

        ## For simplicity, the fit is drawn as a series of points (circles
        ## with radius 1)
        for left_fit_point in zip(left_fit_x, plot_y):
            cv2.circle(output_image, left_fit_point, 1, (255, 255, 0))

        for right_fit_point in zip(right_fit_x, plot_y):
            cv2.circle(output_image, right_fit_point, 1, (255, 255, 0))

        return output_image


    def _lane_fit(self, lane_pixels, image_1x, image_1y, x_conv=1, y_conv=1):
        """ Fit a polynomial to a lane line.

        Args:
        lane_pixels: Pixels defining a lane line.
        image_1x: Nonzero positions along the x-axis.
        image_1y: Nonzero positions along the y-axis.

        x_conv: X-conversion factor (used to scale lane_x prior to fit, default 1)
        y_conv: Y-conversion factor (used to scale lane_y prior to fit, default 1)

        Returns:
        A polynomial fit for the lane line.
        """

        lane_x, lane_y = image_1x[lane_pixels], image_1y[lane_pixels]
        return np.polyfit(lane_y*y_conv, lane_x*x_conv, 2)


    def _window_boundaries(self, wi, centre_x, image_shape):
        """ Compute the boundaries of a window

        Args:
        wi: Row index of the window.
        image_shape: The shape of the input image.
        centre_x: Centre of the window (computed elsewhere)

        Returns:
        Dict with 'x0', 'x1', 'y0', 'y1', containing the lower left
        corner, and the upper right corner of the window, respectively.
        """

        window_height = np.int(np.floor(image_shape[0] / self.n_windows))

        window_y_low = image_shape[0] - (wi + 1) * window_height
        window_y_top = image_shape[0] - wi * window_height

        window  = {'x0': np.int(np.floor(centre_x - self.window_width/2)),
                   'y0': np.int(np.floor(window_y_low)),
                   'x1': np.int(np.floor(centre_x + self.window_width/2)),
                   'y1': np.int(np.floor(window_y_top))}

        return window


    def _init_lane_pixels(self, x0, y0, x1, y1, image_1x, image_1y):
        """ Determine the pixels that are part of a lane line, in a window.

        Args:
        x0: Lower left corner, x
        y0: Lower left corner, y
        x1: Upper left corner, x
        y1: Upper left corner, y
        image_1x: Nonzero pixels, along the x-axis
        image_1y: Nonzero pixels, along the y-axis

        Returns:
        Nonzero pixels within the specified boundaries.
        """
        pixels =  ((image_1x >= x0) & (image_1x < x1) & (image_1y >= y0) & (image_1y < y1))
        return pixels.nonzero()[0]


    def _lane_init(self, binary_image):
        """ Initialize lane line locations

        The lane line procedure requires initialization at the most likely
        point where lane lines will begin. Here, we compute the point in the right
        half of the image, and the point in the left half of the image, where the
        histogram (which will appear to be bimodal, ideally), reaches a maximum.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        Returns:
        A tuple, (x0, x1), corresponding to the locations where the
        left and right lane lines were detected.
        """

        image_shape = binary_image.shape

        hist = np.sum(binary_image[np.int(image_shape[0]/2):,:], axis=0)

        hist_midpoint = np.int(hist.shape[0]/2)
        hist_leftmax_x  = np.argmax(hist[:hist_midpoint])
        hist_rightmax_x = np.argmax(hist[hist_midpoint:]) + hist_midpoint

        return (hist_leftmax_x, hist_rightmax_x)

    def _image_nonzero(self, binary_image):
        """ Get the nonzero elements in an image.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        Returns:
        A tuple, (nonzero x, nonzero y), each is an array indicating nonzero positions.
        """
        image_1 = binary_image.nonzero()
        image_1y = np.array(image_1[0])
        image_1x = np.array(image_1[1])

        return (image_1x, image_1y)

    def _compute_curvature(self, binary_image, lane_pixels, image_1x, image_1y):
        """ Compute the radius of curvature given the left, right fit

        The constants applied here for the  converstion from pixel space to real space are
        those provided by Udacity.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        lane_pixels: Pixel locations believed to be part of lane lines.
        image_1x: Nonzero, along x
        image_1y: Nonzero, along y

        Returns:
        Scalar, curvature.
        """

        y = binary_image.shape[0]

        ym_per_pixels = 30/720
        xm_per_pixels = 3.7/700

        new_fit = self._lane_fit(lane_pixels, image_1x, image_1y, x_conv=xm_per_pixels, y_conv=ym_per_pixels)

        curvature = lambda y, a, b: ((1 + (2*a*y*ym_per_pixels + b)**2)**(3/2)) / np.absolute(2*a)

        return (eval_poly(y*ym_per_pixels, new_fit), curvature(y, new_fit[0], new_fit[1]))
