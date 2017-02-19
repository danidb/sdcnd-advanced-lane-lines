## Sources related to performing a perspective transform on an image.
import cv2

class Perspectifier:
    """ Define and apply a perspective transform.

    Args:
    source_points: Source points for perspective transform. Points defined
    on the original image.

    destination_points: Destination points for perspective transform, points defined
    on the output image.
    """
    def __init__(self, source_points, destination_points):
        self.source_points = source_points
        self.destination_points = destination_points
        self.transformation_matrix = cv2.getPerspectiveTransform(source_points, destination_points)


    def perspectify(self, image):
        """ Transform image perspective using OpenCV functions.

        Args:
        image: The input image, to be transformed.

        Returns:
        Input 'image' with the perspective transformed.
        """

        return cv2.warpPerspective(image, self.transformation_matrix,
                                   image.shape, flags=cv2.INTER_CUBIC)
