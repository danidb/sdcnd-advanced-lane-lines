# Image preprocessing (CLAHE, cropping etc.)
import cv2
import numpy as np

def preproc(image):
    """ Pre-process an image in the lane detection pipeline.

    The pre-processing is as follows. First, CLAHE is applied to the
    input color image. Then, this image is converted to HLS color space, and
    the s-channel is extracted. The same input image is also converted to
    grayscale, and the absolute value of the gradient in the x, y directions are computed.

    Args:
    image: Image to be pre-processed. Must be an RGB color image.

    Returns:
    A tuple with three ndarrays representing the image gradient in the x direction,
    the image gradient in the y direction, and the s-channel of the image, after
    conversion to HLS space. The width/height of the input remain the same.
    """
    image_working = clahe_RGB(image, clipLimit=2, tileGridSize=(4,4))
    image_working = cv2.GaussianBlur(image_working, (3, 3), 0)

    image_ret_s = cv2.cvtColor(image_working, cv2.COLOR_RGB2HLS)
    image_ret_s = image_working[:,:,2]

    image_working = cv2.cvtColor(image_working, cv2.COLOR_RGB2GRAY)

    image_ret_sx = np.absolute(cv2.Sobel(image_working, cv2.CV_64F, 1, 0))
    image_ret_sx = np.uint8(255 * image_ret_sx / np.max(image_ret_sx))

    image_ret_sy = np.absolute(cv2.Sobel(image_working, cv2.CV_64F, 0, 1))
    image_ret_sy = np.uint8(255 * image_ret_sy / np.max(image_ret_sy))

    return (image_ret_sx, image_ret_sy, image_ret_s)

def clahe_RGB(img, clipLimit, tileGridSize):
    """ Apply Contrast-Limited Adaptive Histogram Equalization with OpenCV

    Contrast-Limited Adaptive Histogram Equalization is applied to each
    of the three color channels of an RGB image. The result is returned
    as an RGB image.

    Args:
        img: Input image  should be in RGB colorspace.
        clipLimit: Passed to cv2.createCLAHE
        tileGridSize: Passed to cv2.createCLAHE

    Returns:
        The input image  with CLAHE applied  in RGB
    """

    r, g, b = cv2.split(img)

    img_clahe   = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_clahe_r = img_clahe.apply(r)
    img_clahe_g = img_clahe.apply(g)
    img_clahe_b = img_clahe.apply(b)

    img_ret = cv2.merge((img_clahe_r,  img_clahe_g,  img_clahe_b))

    return(img_ret)
