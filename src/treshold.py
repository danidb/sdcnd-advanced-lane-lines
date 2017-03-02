# Helper sources for thresholding
import numpy as np
import operator

def grad_mag(grad_x, grad_y):
    """ Compute the magnituded of an image gradient, given the gradients in the x and y directions.

    The magnitude is computed as the square root of the sum of grad_x ^ 2 and grad_y ^ 2.
    Prior to return, it is scaled.

    Args:
    grad_x: Image gradient in the x direction.
    grad_y: Image gradient in the y direction.

    Returns:
    An ndarray of the same shape as grad_x and grad_y, containing the magnitued of the
    gradient, as described above.
    """

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = (magnitude/(np.max(magnitude)/255)).astype(np.uint8)

    return magnitude


def grad_dir(grad_x, grad_y):
    """ Compute the directoin of an image gradient, given the gradients in the x and y direction.

    Args:
    grad_x: Image gradient in the x direction.
    grad_y: Image gradient in the y direction.

    Returns:
    An ndarray of the same shape as grad_x and grad_y, containing the direction of the
    image gradient.
    """

    return np.arctan2(np.absolute(grad_y), np.absolute(grad_x))

def bithreshold(x, lower_bound, upper_bound, lower_op, upper_op):
    """ Helper function for applying simple upper and lower bound

    Args:
    x: Values to which the treshold is applied.
    lower_bound: Lower bound of the threshold.
    upper_bound: Upper bound of the threshold.
    lower_op: Operator for lower bound - a string, with the operator symbol.
    upper_op: Operator for upper bound - a string, with the operator symbol.

    Returns:
    Values, 0,1: 0 outside the threshold, 1 within.
    """

    allowed_operators = {'<' : operator.lt,
                         '>' : operator.gt,
                         '<=': operator.le,
                         '>=': operator.ge}

    if lower_op in allowed_operators.keys() and upper_op in allowed_operators.keys():

        lop = allowed_operators[lower_op]
        uop = allowed_operators[upper_op]

        ret = np.zeros_like(x)
        ret[lop(x, lower_bound) & uop(x, upper_bound)] = 1
        ret = np.array(ret, dtype=np.uint8)

    else:

        print("ERROR: Inavlid error, returning None.")
        ret = None

    return ret
