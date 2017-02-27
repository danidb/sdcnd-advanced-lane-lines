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
