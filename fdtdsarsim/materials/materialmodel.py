"""
The materialmodel module defines the MaterialModel class which is the object
type that all material iterators/generators must produce.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MaterialModel:
    """ Dataclass for storing the output of a material generator.
    """
    relative_permittivity: np.ndarray
    conductivity: np.ndarray
    cell_size: float  #: Side length of a cubic Yee cell [m]
    name: str = 'results'
    """
    The name of the model is used for storing the simulation results. Duplicate
    names in a single experiment will cause the results files to overwrite each
    other.
    """

    metadata: dict = None
    """
    The material metadata is a dictionary that can contain numbers, numpy
    arrays or strings which describe the material. These are then stored in
    the results file and can be used for post-processing.
    """
