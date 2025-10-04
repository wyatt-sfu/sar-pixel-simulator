import numpy as np


def round_to_odd(input):
    """
    Return the input value rounded to an odd integer. This is used for
    configuring the geometry of the simulation space.
    """
    if input < 0:
        raise ValueError('Only use this with positive values for now')
    output = np.fix(input).item()
    rem = np.remainder(output, 2).item()
    if rem == 0:
        output += 1
    return int(output)
