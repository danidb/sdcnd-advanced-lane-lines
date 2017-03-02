# Various utility methods
import os

def abswalk(path):
    """ Get the absolute paths from a walk down a directory tree.

    Args:
    path: Path to the top level directory.

    Returns:
    List of absolute paths.
    """
    paths = []
    for this_dir, _,names in os.walk(path):
        paths += [os.path.abspath(os.path.join(this_dir, name)) for name in names]

    return paths

def eval_poly(x, fit):
    """ Evaluate a polynomial specifed in a numpy polyfit

    Args:
    x: Value for which the polynomial must be evaluated.
    fit: Numpy polynomial fit.

    Returns:
    Value of evaulauting the polynomial 'fit' at x.
    """
    degree = len(fit)-1
    return sum([(x ** (degree - i)) * fit[i] for i in range(degree + 1)])

def exp_smooth(x, smoothed_previous, gamma):
    """ Exponentially smooth a value

    Args:
    x: Value to be smoothed
    smoothed_previous: Previous smoothed value.
    gamma: Smoothing parameter.

    Returns:
    gamma * x + (1- gamma)* smothed_previous
    """

    return gamma*x + (1-gamma)*smoothed_previous
