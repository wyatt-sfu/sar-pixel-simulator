"""
This module contains functions for creating rough surface profiles.
"""
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import numba

from fdtdsarsim.utils.plotting.materials import plot_rough_surface_debug


def gaussian_correlated_surface(nx, ny, dx, corr_len_x, corr_len_y, height_rms,
                                seed=None, rabus_compat=False,
                                debug_plots=False):
    """
    Create a 2D surface height array with Gaussian autocorrelation.

    [1] The Importance of Soil Moisture and Soil Structurefor InSAR Phase and
        Backscatter, as Determined by FDTD Modeling. Rabus, Wehn, Nolan.

    Args:
        nx (int): Number of sample points in X direction
        ny (int): Number of sample points in Y direction
        dx (float): Distance between samples (assumed square pixels) [m]
        corr_len_x (float): Correlation length in X direction [m]
        corr_len_y (float): Correlation length in Y direction [m]
        height_rms (float): RMS height of the surface [m]
        seed (int): Optional seed for random number generator
        rabus_compat (bool): Generate a surface compatible with [1]
        debug_plots (bool): Create debug plots

    Returns:
        np.ndarray: Numpy array of surface height with shape (ny, nx).
    """
    orig_corr_len_x = corr_len_x
    orig_corr_len_y = corr_len_y

    if rabus_compat:
        corr_len_x /= np.sqrt(2.0)
        corr_len_y /= np.sqrt(2.0)

    # Create a Gaussian kernel
    kernel_size_x = int(6.0 * round(corr_len_x / dx))
    kernel_size_y = int(6.0 * round(corr_len_y / dx))
    kernel_x = np.arange(-kernel_size_x // 2, kernel_size_x // 2) * dx
    kernel_y = np.arange(-kernel_size_y // 2, kernel_size_y // 2) * dx
    xx, yy = np.meshgrid(kernel_x, kernel_y)
    kernel = np.exp(-(xx**2 / corr_len_x**2 + yy**2 / corr_len_y**2))

    # Generate white noise
    rng = np.random.default_rng(seed=seed)
    noise_shape = (ny + kernel_size_y - 1, nx + kernel_size_x - 1)
    white_noise = rng.normal(0, 1, size=noise_shape)

    # Convolve the noise with the kernel
    gauss_corr = scipy.signal.fftconvolve(white_noise, kernel, mode='valid')

    # Normalize the data
    rough_surf = gauss_corr - np.mean(gauss_corr)
    rough_surf /= np.std(rough_surf)
    rough_surf *= height_rms

    if debug_plots:
        plot_rough_surface_debug(nx, ny, dx, orig_corr_len_x, orig_corr_len_y,
                                 height_rms, rough_surf, kernel)
        plt.show()

    return rough_surf


@numba.jit(nopython=True, parallel=True)
def gaussian_correlated_surface_fast(nx, ny, dx, corr_len_x, corr_len_y,
                                     height_rms, rand_gen, rabus_compat=False):
    """
    Create a 2D surface height array with Gaussian autocorrelation.
    This is the fast version of the function which uses numba but cannot
    perform debug plotting.

    [1] The Importance of Soil Moisture and Soil Structurefor InSAR Phase and
        Backscatter, as Determined by FDTD Modeling. Rabus, Wehn, Nolan.

    Args:
        nx (int): Number of sample points in X direction
        ny (int): Number of sample points in Y direction
        dx (float): Distance between samples (assumed square pixels) [m]
        corr_len_x (float): Correlation length in X direction [m]
        corr_len_y (float): Correlation length in Y direction [m]
        height_rms (float): RMS height of the surface [m]
        rand_gen (np.random.Generator): Random number generator
        rabus_compat (bool): Generate a surface compatible with [1]

    Returns:
        np.ndarray: Numpy array of surface height with shape (ny, nx).
    """
    if rabus_compat:
        corr_len_x /= np.sqrt(2.0)
        corr_len_y /= np.sqrt(2.0)

    corr_len_x = np.float32(corr_len_x)
    corr_len_y = np.float32(corr_len_y)
    height_rms = np.float32(height_rms)

    # Create a Gaussian kernel
    kernel_size_x = int(6.0 * round(corr_len_x / dx))
    kernel_size_y = int(6.0 * round(corr_len_y / dx))
    kernel_xvals = np.arange(-kernel_size_x // 2, kernel_size_x // 2) * dx
    kernel_xvals = kernel_xvals.astype(np.float32)
    kernel_yvals = np.arange(-kernel_size_y // 2, kernel_size_y // 2) * dx
    kernel_yvals = kernel_yvals.astype(np.float32)
    kernel_y = np.exp(-(kernel_yvals**2 / corr_len_y**2))
    kernel_x = np.exp(-(kernel_xvals**2 / corr_len_x**2))

    # Generate white noise
    noise_shape = (ny, nx)
    white_noise = rand_gen.standard_normal(size=noise_shape, dtype=np.float32)

    # Convolve the noise with the kernel
    # gauss_corr = np.zeros(noise_shape, dtype=np.float32)
    for y in numba.prange(white_noise.shape[0]):
        tmp_x = np.convolve(white_noise[y, :], kernel_x)
        white_noise[y, :] = tmp_x[0:nx]
    for x in numba.prange(white_noise.shape[1]):
        tmp_y = np.convolve(white_noise[:, x], kernel_y)
        white_noise[:, x] = tmp_y[0:ny]

    # Normalize the data
    rough_surf = white_noise - np.float32(np.mean(white_noise))
    rough_surf /= np.float32(np.std(rough_surf))
    rough_surf *= height_rms
    return rough_surf


def two_layer_gaussian_correlated_surface(
        nx, ny, dx, corr_len_x, corr_len_y, height_rms, inter_layer_corr,
        seed=None, rabus_compat=False, debug_plots=False):
    """
    Create a two layer surface height array. The surfaces have a spatial
    Gaussian autocorrelation. The layers are correlated based on the
    inter_layer_corr argument.

    [1] The Importance of Soil Moisture and Soil Structurefor InSAR Phase and
        Backscatter, as Determined by FDTD Modeling. Rabus, Wehn, Nolan.

    Args:
        nx (int): Number of sample points in X direction
        ny (int): Number of sample points in Y direction
        dx (float): Distance between samples (assumed square pixels) [m]
        corr_len_x (float): Correlation length in X direction [m]
        corr_len_y (float): Correlation length in Y direction [m]
        height_rms (float): RMS height of the surface [m]
        inter_layer_corr (float): Correlation coefficient between layers
            (-1 to 1).
        seed (int): Optional seed for random number generator
        rabus_compat (bool): Generate a surface compatible with [1]
        debug_plots(bool): Create debug plots

    Returns:
        np.ndarray: Numpy array of surface height with shape (2, ny, nx).
    """
    orig_corr_len_x = corr_len_x
    orig_corr_len_y = corr_len_y

    if rabus_compat:
        corr_len_x /= np.sqrt(2.0)
        corr_len_y /= np.sqrt(2.0)

    nlayers = 2

    if inter_layer_corr >= 1.0:
        inter_layer_corr = 1.0 - 1e-5

    # Create a Gaussian kernel
    kernel_size_x = int(6.0 * round(corr_len_x / dx))
    kernel_size_y = int(6.0 * round(corr_len_y / dx))
    kernel_x = np.arange(-kernel_size_x // 2, kernel_size_x // 2) * dx
    kernel_y = np.arange(-kernel_size_y // 2, kernel_size_y // 2) * dx
    xx, yy = np.meshgrid(kernel_x, kernel_y)
    kernel = np.exp(-(xx**2 / corr_len_x**2 + yy**2 / corr_len_y**2))

    # Generate white noise
    rng = np.random.default_rng(seed=seed)
    noise_shape = (ny + kernel_size_y - 1, nx + kernel_size_x - 1, nlayers)
    white_noise = rng.normal(0, 1, size=noise_shape)

    # Inter layer correlation
    covariance = np.zeros((2, 2))
    covariance[0, 0] = 1  # Both layers have the same variance
    covariance[1, 1] = covariance[0, 0]
    covariance[0, 1] = inter_layer_corr
    covariance[1, 0] = covariance[0, 1]
    L = np.linalg.cholesky(covariance)
    layers = correlate_layers(white_noise, L)
    layers = np.transpose(layers, [2, 0, 1])

    # Convolve the noise with the kernel
    gauss_corr = np.zeros((nlayers, ny, nx))
    rough_surf = np.zeros((nlayers, ny, nx))
    for i in range(nlayers):
        gauss_corr[i, :, :] = scipy.signal.convolve2d(
            layers[i, :, :], kernel, mode='valid')
        rough_surf[i, ...] = gauss_corr[i, ...] - np.mean(gauss_corr[i, ...])
        rough_surf[i, ...] /= np.std(rough_surf[i, ...])
        rough_surf[i, ...] *= height_rms

    if debug_plots:
        plot_rough_surface_debug(nx, ny, dx, orig_corr_len_x, orig_corr_len_y,
                                 height_rms, rough_surf[0, :, :], kernel)
        plot_rough_surface_debug(nx, ny, dx, orig_corr_len_x, orig_corr_len_y,
                                 height_rms, rough_surf[1, :, :], kernel)
        plt.show()

    return rough_surf


@numba.njit
def correlate_layers(white_noise, cholesky):
    ny = white_noise.shape[0]
    nx = white_noise.shape[1]
    layers = np.zeros_like(white_noise)
    for y in range(ny):
        for x in range(nx):
            layers[y, x, :] = cholesky @ white_noise[y, x, :]

    return layers
