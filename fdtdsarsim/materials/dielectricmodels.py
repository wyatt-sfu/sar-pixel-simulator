"""
The dielectric_models module defines some useful models of the dielectric
properties of different materials in the microwave spectrum. Each model cites
its source and will raise an exception if it is used in a way which violates
the assumptions laid out in the source material.
"""
import numpy as np
import numba


class HallikainenSoilModel:
    def __init__(self):
        """
        This class computes the dielectric permittivity for wet soils using
        the Hallikainen model (see citation below). The polynomial coefficients
        are found in Table 2 of the paper referenced below.

        References:
        M. Hallikainen, F. Ulaby, M. Dobson, M. El-rayes, and L.-kun Wu,
        “Microwave dielectric behavior of wet soil-part 1: Empirical models and
        experimental observations,” IEEE Transactions on Geoscience and Remote
        Sensing, 1985.
        """
        self.freq_table = np.array(
            [1.4, 4, 6, 8, 10, 12, 14, 16, 18]) * 1.0e9

        self.a_coeff_re = np.array([
            [2.862, -0.012, 0.001],
            [2.927, -0.012, -0.001],
            [1.993, 0.002, 0.015],
            [1.997, 0.002, 0.018],
            [2.502, -0.003, -0.003],
            [2.200, -0.001, 0.012],
            [2.301, 0.001, 0.009],
            [2.237, 0.002, 0.009],
            [1.912, 0.007, 0.021]
        ])

        self.a_coeff_im = np.array([
            [0.356, -0.003, -0.008],
            [0.004, 0.001, 0.002],
            [-0.123, 0.002, 0.003],
            [-0.201, 0.003, 0.003],
            [-0.070, 0.000, 0.001],
            [-0.142, 0.001, 0.003],
            [-0.096, 0.001, 0.002],
            [-0.027, -0.001, 0.003],
            [-0.071, 0.000, 0.003]
        ])

        self.b_coeff_re = np.array([
            [3.803, 0.462, -0.341],
            [5.505, 0.371, 0.062],
            [38.086, -0.176, -0.633],
            [25.579, -0.017, -0.412],
            [10.101, 0.221, -0.004],
            [26.473, 0.013, -0.523],
            [17.918, 0.084, -0.282],
            [15.505, 0.076, -0.217],
            [29.123, -0.190, -0.545]
        ])

        self.b_coeff_im = np.array([
            [5.507, 0.044, -0.002],
            [0.951, 0.005, -0.010],
            [7.502, -0.058, -0.116],
            [11.266, -0.085, -0.155],
            [6.620, 0.015, -0.081],
            [11.868, -0.059, -0.225],
            [8.583, -0.005, -0.153],
            [6.179, 0.074, -0.086],
            [6.938, 0.029, -0.128]
        ])

        self.c_coeff_re = np.array([
            [119.006, -0.500, 0.633],
            [114.826, -0.389, -0.547],
            [10.720, 1.256, 1.522],
            [39.793, 0.723, 0.941],
            [77.482, -0.061, -0.135],
            [34.333, 0.284, 1.062],
            [50.149, 0.012, 0.387],
            [48.260, 0.168, 0.289],
            [6.960, 0.822, 1.195]
        ])

        self.c_coeff_im = np.array([
            [17.753, -0.313, 0.206],
            [16.759, 0.192, 0.290],
            [2.942, 0.452, 0.543],
            [0.194, 0.584, 0.581],
            [21.578, 0.293, 0.332],
            [7.817, 0.570, 0.801],
            [28.707, 0.297, 0.357],
            [34.126, 0.143, 0.206],
            [29.945, 0.275, 0.377]
        ])

    def get_epsilon(self, mv: float, sand_frac: float, clay_frac: float,
                    freq: float):
        """ Compute the permittivity for the material properties specified.

        Args:
            mv (float): Volumetric soil moisture (0-1)
            sand_frac (float): Fractional sand content of dry soil (0-1)
            clay_frac (float): Fractional clay content of dry soil (0-1)
            freq (float): Frequency of interest [Hz]

        Returns:
            complex: Epsilon, the complex dielectric constant (unitless)
        """
        # Check that the arguments are inside the valid bounds
        if mv < 0.0 or mv > 1.0:
            raise ValueError('Volumetric moisture out of bounds (0 < mv < 1)')
        if sand_frac < 0.0 or sand_frac > 1.0:
            raise ValueError('Sand fraction out of bounds (0 < sand_frac < 1')
        if clay_frac < 0.0 or clay_frac > 1.0:
            raise ValueError('Clay fraction out of bounds (0 < clay_frac < 1)')
        if freq < 1.4e9 or freq > 18.0e9:
            raise ValueError('Frequency out of bounds (1.4 GHz < freq < 18 '
                             'GHz)')

        vec = np.array([1.0, sand_frac, clay_frac])
        a_real = np.dot(self.a_coeff_re, vec)
        a_imag = np.dot(self.a_coeff_im, vec)
        b_real = np.dot(self.b_coeff_re, vec)
        b_imag = np.dot(self.b_coeff_im, vec)
        c_real = np.dot(self.c_coeff_re, vec)
        c_imag = np.dot(self.c_coeff_im, vec)

        eps_real = a_real + (b_real * mv) + (c_real * (mv ** 2))
        eps_imag = a_imag + (b_imag * mv) + (c_imag * (mv ** 2))

        # Interpolate to the desired frequency
        epsilon = np.interp(freq, self.freq_table, eps_real) \
            - 1j * np.interp(freq, self.freq_table, eps_imag)
        return epsilon


"""
Unfortunately this is super ugly but it helps speed up the numba jit
compiled version to have these defined outside the function.
"""
freq_table = np.array(
[1.4, 4, 6, 8, 10, 12, 14, 16, 18]) * 1.0e9

a_coeff_re = np.array([
    [2.862, -0.012, 0.001],
    [2.927, -0.012, -0.001],
    [1.993, 0.002, 0.015],
    [1.997, 0.002, 0.018],
    [2.502, -0.003, -0.003],
    [2.200, -0.001, 0.012],
    [2.301, 0.001, 0.009],
    [2.237, 0.002, 0.009],
    [1.912, 0.007, 0.021]
])

a_coeff_im = np.array([
    [0.356, -0.003, -0.008],
    [0.004, 0.001, 0.002],
    [-0.123, 0.002, 0.003],
    [-0.201, 0.003, 0.003],
    [-0.070, 0.000, 0.001],
    [-0.142, 0.001, 0.003],
    [-0.096, 0.001, 0.002],
    [-0.027, -0.001, 0.003],
    [-0.071, 0.000, 0.003]
])

b_coeff_re = np.array([
    [3.803, 0.462, -0.341],
    [5.505, 0.371, 0.062],
    [38.086, -0.176, -0.633],
    [25.579, -0.017, -0.412],
    [10.101, 0.221, -0.004],
    [26.473, 0.013, -0.523],
    [17.918, 0.084, -0.282],
    [15.505, 0.076, -0.217],
    [29.123, -0.190, -0.545]
])

b_coeff_im = np.array([
    [5.507, 0.044, -0.002],
    [0.951, 0.005, -0.010],
    [7.502, -0.058, -0.116],
    [11.266, -0.085, -0.155],
    [6.620, 0.015, -0.081],
    [11.868, -0.059, -0.225],
    [8.583, -0.005, -0.153],
    [6.179, 0.074, -0.086],
    [6.938, 0.029, -0.128]
])

c_coeff_re = np.array([
    [119.006, -0.500, 0.633],
    [114.826, -0.389, -0.547],
    [10.720, 1.256, 1.522],
    [39.793, 0.723, 0.941],
    [77.482, -0.061, -0.135],
    [34.333, 0.284, 1.062],
    [50.149, 0.012, 0.387],
    [48.260, 0.168, 0.289],
    [6.960, 0.822, 1.195]
])

c_coeff_im = np.array([
    [17.753, -0.313, 0.206],
    [16.759, 0.192, 0.290],
    [2.942, 0.452, 0.543],
    [0.194, 0.584, 0.581],
    [21.578, 0.293, 0.332],
    [7.817, 0.570, 0.801],
    [28.707, 0.297, 0.357],
    [34.126, 0.143, 0.206],
    [29.945, 0.275, 0.377]
])

@numba.njit
def hallikainen_soil_model(mv: float, sand_frac: float, clay_frac: float,
                           freq: float):
    """
    This function computes the dielectric permittivity for wet soils using
    the Hallikainen model (see citation below). The polynomial coefficients
    are found in Table 2 of the paper referenced below.

    Note: Use this function over the HallikainenSoilModel class when you
    want to compute the soil properties inside of a numba jit compiled
    function.

    References:
    M. Hallikainen, F. Ulaby, M. Dobson, M. El-rayes, and L.-kun Wu,
    “Microwave dielectric behavior of wet soil-part 1: Empirical models and
    experimental observations,” IEEE Transactions on Geoscience and Remote
    Sensing, 1985.

    Args:
        mv (float): Volumetric soil moisture (0-1)
        sand_frac (float): Fractional sand content of dry soil (0-1)
        clay_frac (float): Fractional clay content of dry soil (0-1)
        freq (float): Frequency of interest [Hz]

    Returns:
        complex: Epsilon, the complex dielectric constant (unitless)
    """
    vec = np.array([1.0, sand_frac, clay_frac])
    a_real = np.dot(a_coeff_re, vec)
    a_imag = np.dot(a_coeff_im, vec)
    b_real = np.dot(b_coeff_re, vec)
    b_imag = np.dot(b_coeff_im, vec)
    c_real = np.dot(c_coeff_re, vec)
    c_imag = np.dot(c_coeff_im, vec)

    eps_real = a_real + (b_real * mv) + (c_real * (mv ** 2))
    eps_imag = a_imag + (b_imag * mv) + (c_imag * (mv ** 2))

    # Interpolate to the desired frequency
    epsilon = np.interp(freq, freq_table, eps_real) \
        - 1j * np.interp(freq, freq_table, eps_imag)
    return epsilon
