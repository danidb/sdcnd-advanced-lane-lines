import cv2
import os
import numpy as np
from scipy import misc
import progressbar

class Calibrator:
    """ A camera calibrator, based on chessboard callibration images.

    Note that for the purposes of this exercise, we forego all manner of sensible
    input checks..."we're all adults" right?

    Args:

    calibration_image_directory: Directory of calibration (chessboard) iamges.

    chessboard_size: Size of chessboards in the calibration images.

    calibration_image_paths: List of filepaths for chessboard images to be used for calibration.

    """
    def __init__(self, chessboard_size, calibration_image_paths):

        self.chessboard_size = chessboard_size
        self.calibration_image_paths = calibration_image_paths
        self.test_image_paths = None


        # Initialize instance data to be prepared in __build_calibrator
        self.mtx  = None
        self.dist = None

        self.__build_calibrator()


    def __build_calibrator(self):
        """ Prepare the calibrator.

        Reads in calibration images and prepares this calibrator, using
        openCV's findChessboardcorners. This method updates self.mtx
        and self.dist.

        Args:
        None. Operates on instance data.

        Returns:
        None. Assigns instance data.
        """

        cb_nrows = self.chessboard_size[0]
        cb_ncols = self.chessboard_size[1]

        object_point_set = np.zeros((cb_nrows * cb_ncols, 3), dtype=np.float32)
        object_point_set[:,:2] = np.mgrid[0:cb_nrows, 0:cb_ncols].T.reshape(-1, 2)

        image_points = []
        object_points = []

        ## Collect corners/object points for each image, they are all used together
        ## to produce the final calibration matrix.
        pbar = progressbar.ProgressBar(max_value=len(self.calibration_image_paths))
        pbar.start()

        for i, image_path in enumerate(self.calibration_image_paths):

            image = misc.imread(image_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners  = cv2.findChessboardCorners(image_gray, self.chessboard_size, None)

            if ret:
                image_out = cv2.drawChessboardCorners(image, self.chessboard_size, corners, ret)
                image_points += [corners]
                object_points += [object_point_set]

            pbar.update(i)

        pbar.finish()
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(object_points, image_points, image_gray.shape[::-1], None, None)


    def undistort(self, image):
        """ Undistort an image using this calibrator.

        Args:
        image: Image to be un-distorted.

        Returns:
        Undistorted image.
        """

        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
