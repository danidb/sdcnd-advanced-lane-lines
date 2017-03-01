import numpy as np
import cv2
from util import eval_poly

class LaneOMatic:
    """ The 'Lane-O-Matic' class defines a lane line detection procedure.

    The procedure defined within this class, relies on pre-processed input, and
    the original source image. This class contains facilities for detecting lane
    lines via a sliding window.

    For videos, the lane line detection procedure can be smoothed by taking into
    account the estimates for the last N frames. Exponential smoothing is applied.

    Args:
    smooth: Should smoothing be applied?

    smoother_decay: Decay parameter (exponential, should be in [0,1]) for smoothing.

    n_windows: Number of sliding windows.

    window_width: Width of windows. Consider a mass of lane line pixels to be the centre,
    and the window will extend +/- window_width/2 about that centre.

    window_threshold:
    """

    def __init__(self, smooth=True, smoother_gamma=0.3, n_windows=9, window_width=100, window_threshold=50):
        self.smooth = smooth
        self.smoother_gamma = smoother_gamma
        self.n_windows = n_windows
        self.window_width = window_width
        self.window_threshold = window_threshold

        self.previous_fit = None


    def detect_lanes(self, binary_image) :
        """ Detect the lane lines in the provided binary image, and return the polynomial fit.

        Args:
        binappry_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        Returns:
        Output image, with detection information superimposed.
        """

        if self.previous_fit:
            fit, output_image = self.secondary_detection(binary_image)
        else:
            fit, output_image = self.primary_detection(binary_image)

        if self.smooth:
            self.previous_fit = fit

        return output_image

    def secondary_detection(self, binary_image):
        """ Detection for frames of video after the first.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        Returns:
        None. Updates instance data, draws output image.
        """

        imabe_1x, image_1y = self._image_nonzero(binary_image)

        prev_left_fit, prev_right_fit = self.previous_fit

        left_lane = ((image_1x > eval_poly(image_1y, prev_left_fit) - margin) &
                     (image_1x < eval_poly(image_1y, prev_left_fit) + margin))
        right_lane = ((image_1x > eval_poly(image_1y, prev_right_fit) - margin) &
                      (image_1x < eval_poly(image_1y, prev_right_fit) + margin))

        left_fit = self._lane_fit(left_lane, image_1x, image_1y)
        right_fit = self._lane_fit(right_lane, image_1x, image_1y)

        return (left_fit, right_fit)


    def primary_detection(self, binary_image):
        """ Detect the lane lines in an image, or the first frame of a video.

        The lane line procedure is based on a sliding window. The initial
        phase of the procedure uses a histogram to select two starting points
        for the columns of windows. This method is the first step to detecting
        lane lines in a video, or the only step required to detect lines in a
        single image.

        Args:
        binary_image: The image on which the lane lines are to be detected. Note that all pre-processing
        should have been applied to the image prior to this step. The image must be binary, and
        contain only values in {0, 1}.

        Returns:
        None. Updates instance data, draws output image.
        """


        image_1x, image_1y = self._image_nonzero(binary_image)

        left_x_init, right_x_init = self._lane_init(binary_image)
        left_x, right_x = left_x_init, right_x_init

        left_lanes, right_lanes = [], []
        left_windows, right_windows = [], []

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


        left_fit  = self._lane_fit(left_lanes, image_1x, image_1y)
        right_fit = self._lane_fit(right_lanes, image_1x, image_1y)

        output_image = self._draw_lane_windows(binary_image, left_windows, np.concatenate(left_lanes), left_fit,
                                               right_windows, np.concatenate(right_lanes), right_fit, image_1x, image_1y)

        return ((left_fit, right_fit), output_image)

    def _lane_fit(self, lane_pixels, image_1x, image_1y):
        """ Fit a polynomial to a lane line.

        Args:
        lane_pixels: Pixels defining a lane line.
        image_1x: Nonzero positions along the x-axis.
        image_1y: Nonzero positions along the y-axis.

        Returns:
        A polynomial fit for the lane line.
        """

        lane_x, lane_y = image_1x[np.concatenate(lane_pixels)], image_1y[np.concatenate(lane_pixels)]
        return np.polyfit(lane_y, lane_x, 2)


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

        hist = np.sum(binary_image[image_shape[0]/2:,:], axis=0)

        hist_midpoint = np.int(hist.shape[0]/2)
        hist_leftmax_x  = np.argmax(hist[:hist_midpoint])
        hist_rightmax_x = np.argmax(hist[hist_midpoint:]) + hist_midpoint

        return (hist_leftmax_x, hist_rightmax_x)

    def _draw_lane_windows(self, binary_image, left_windows, left_pixels,  left_fit,
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
        image_1x, image_1y = self._image_nonzero(binary_image)

        plot_y = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0], dtype=np.uint16)

        left_fit_x = np.array(eval_poly(plot_y, left_fit), dtype=np.uint8)
        right_fit_x = np.array(eval_poly(plot_y, right_fit), dtype=np.uint8)

        ## The lane line pixels are filled with red, and blue to easily
        ## differentiate between the two lanes. Below the following two lines,
        ## windows are outlined in green.
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

        for left_fit_point in zip(left_fit_x, plot_y):
            cv2.circle(output_image, left_fit_point, 1, (255, 255, 0))

        for right_fit_point in zip(right_fit_x, plot_y):
            cv2.circle(output_image, right_fit_point, 1, (255, 255, 0))

        return output_image

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
