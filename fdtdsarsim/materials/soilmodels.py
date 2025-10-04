"""
This module defines soil specific material models.
"""
from typing import Sequence
import numpy as np
import numba
from scipy.constants import epsilon_0

from fdtdsarsim.parameters import ParameterCollection
from .materialmodel import MaterialModel
from .roughsurface import (gaussian_correlated_surface,
                           two_layer_gaussian_correlated_surface)
from .dielectricmodels import HallikainenSoilModel, hallikainen_soil_model


def one_layer_gaussian(params: ParameterCollection, moistures: Sequence[float],
                       sand_frac: float, clay_frac: float, height_rms: float,
                       corr_len: float, mean_depth: float, count: int,
                       seed: int, grooved: bool = False,
                       rabus_compat: bool = False):
    """
    Yields a series of MaterialModel's containing a dielectric with a Gaussian
    correlated rough surface. The dielectric properties are computed using
    the Hallikainen model.
    """
    nx = params.geom_params.size_x
    ny = params.geom_params.size_y
    nz = params.geom_params.size_z_material
    dtype = params.processing_params.datatype
    dx = params.geom_params.cell_size
    max_height = nz * dx
    mean_depth_cells = mean_depth / dx
    freq = params.inc_beam_params.frequency

    if mean_depth > max_height:
        raise ValueError('Requested mean soil depth is greater than the max '
                         'height of the simulator space.')
    soil_model = HallikainenSoilModel()

    # We need a small numba jit'ed function for filling in the material
    # arrays to be fast
    @numba.njit
    def fill_in_material(_eps_arr, _sigma_arr, _surf_height, _epsilon_r,
                         _conductivity):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if z <= _surf_height[y, x]:
                        _eps_arr[z, y, x] = _epsilon_r
                        _sigma_arr[z, y, x] = _conductivity

    for i in range(count):
        surf_height = gaussian_correlated_surface(
            nx, ny, dx, corr_len, corr_len, height_rms, seed=seed + i,
            rabus_compat=rabus_compat)
        if grooved:
            groove_profile = surf_height[0, :]
            groove_profile = groove_profile[np.newaxis, :]
            surf_height = np.repeat(groove_profile, ny, axis=0)
        surf_height /= dx
        surf_height += mean_depth_cells
        surf_height = np.rint(surf_height).astype(np.int32)

        for moisture in moistures:
            eps_cplx = soil_model.get_epsilon(
                moisture, sand_frac, clay_frac, freq)
            epsilon_r = np.real(eps_cplx)
            conductivity = np.abs(np.imag(eps_cplx)) * epsilon_0 \
                * 2.0 * np.pi * freq

            metadata = {
                'moisture': moisture,
                'sand_frac': sand_frac,
                'clay_frac': clay_frac,
                'epsilon_r': epsilon_r,
                'conductivity': conductivity,
                'height_rms': height_rms,
                'mean_depth': mean_depth,
                'mean_depth_cells': mean_depth_cells,
                'corr_len': corr_len
            }

            name = f'mv_{moisture:.2f}_num_{i}'

            # Create arrays of vacuum material
            eps = np.ones((nz, ny, nx), dtype=dtype)
            sigma = np.zeros((nz, ny, nx), dtype=dtype)
            fill_in_material(eps, sigma, surf_height, epsilon_r, conductivity)
            m = MaterialModel(eps, sigma, dx, name, metadata)
            yield m


def two_layer_gaussian(params: ParameterCollection, top_moisture: float,
                       bottom_moistures: Sequence[float], sand_frac: float,
                       clay_frac: float, height_rms: float, corr_len: float,
                       inter_layer_corr: float, mean_total_depth: float,
                       top_layer_height: float, count: int, seed: int,
                       grooved: bool = False, rabus_compat: bool = False):
    """
    Yields a series of MaterialModel's containing a two layer dielectric with
    Gaussian correlated rough surfaces. The user can specify the correlation
    between the two layers. The dielectric properties are computed using
    the Hallikainen model.
    """
    nx = params.geom_params.size_x
    ny = params.geom_params.size_y
    nz = params.geom_params.size_z_material
    dtype = params.processing_params.datatype
    dx = params.geom_params.cell_size
    max_height = nz * dx
    total_depth_cells = mean_total_depth / dx
    top_layer_cells = top_layer_height / dx
    freq = params.inc_beam_params.frequency

    if mean_total_depth > max_height:
        raise ValueError('Requested mean soil depth is greater than the max '
                         'height of the simulator space.')

    if top_layer_cells > total_depth_cells:
        raise ValueError('Top layer of soil is thicker than entire soil model')

    soil_model = HallikainenSoilModel()

    # We need a small numba jit'ed function for filling in the material
    # arrays to be fast
    @numba.njit
    def fill_in_material(_eps_arr, _sigma_arr, _surf_height, _epsilon_r,
                         _conductivity):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if z <= _surf_height[0, y, x]:
                        _eps_arr[z, y, x] = _epsilon_r[0]
                        _sigma_arr[z, y, x] = _conductivity[0]
                    elif z <= _surf_height[1, y, x]:
                        _eps_arr[z, y, x] = _epsilon_r[1]
                        _sigma_arr[z, y, x] = _conductivity[1]

    # Top layer soil properties
    eps_cplx_top = soil_model.get_epsilon(top_moisture, sand_frac, clay_frac,
                                          freq)
    epsilon_r_top = np.real(eps_cplx_top)
    cond_top = np.abs(np.imag(eps_cplx_top)) * epsilon_0 * 2.0 * np.pi * freq

    for i in range(count):
        surf_height = two_layer_gaussian_correlated_surface(
            nx, ny, dx, corr_len, corr_len, height_rms, inter_layer_corr,
            seed + i, rabus_compat=rabus_compat)
        if grooved:
            groove_profile = surf_height[:, 0, :]
            groove_profile = groove_profile[:, np.newaxis, :]
            surf_height = np.repeat(groove_profile, ny, axis=1)

        surf_height /= dx
        surf_height += total_depth_cells
        surf_height[0, :, :] -= top_layer_cells
        surf_height = np.rint(surf_height).astype(np.int32)

        for moisture in bottom_moistures:
            # Bottom layer soil properties
            eps_cplx_bot = soil_model.get_epsilon(moisture, sand_frac,
                                                  clay_frac, freq)
            epsilon_r_bot = np.real(eps_cplx_bot)
            cond_bot = np.abs(np.imag(eps_cplx_bot)) * epsilon_0 \
                * 2.0 * np.pi * freq

            # Put together the layer properties [bottom, top]
            epsilon_r = np.asarray([epsilon_r_bot, epsilon_r_top])
            conductivity = np.asarray([cond_bot, cond_top])

            metadata = {
                'top_moisture': top_moisture,
                'bottom_moisture': moisture,
                'sand_frac': sand_frac,
                'clay_frac': clay_frac,
                'epsilon_r': epsilon_r,
                'conductivity': conductivity,
                'height_rms': height_rms,
                'mean_total_depth': mean_total_depth,
                'total_depth_cells': total_depth_cells,
                'top_layer_height': top_layer_height,
                'top_layer_cells': top_layer_cells,
                'inter_layer_corr': inter_layer_corr,
                'corr_len': corr_len
            }

            name = f'mv_{moisture:.2f}_num_{i}'

            # Create arrays of vacuum material
            eps = np.ones((nz, ny, nx), dtype=dtype)
            sigma = np.zeros((nz, ny, nx), dtype=dtype)
            fill_in_material(eps, sigma, surf_height, epsilon_r, conductivity)
            m = MaterialModel(eps, sigma, dx, name, metadata)
            yield m


def gradient_gaussian(params: ParameterCollection, top_moisture: float,
                      bottom_moistures: Sequence[float], sand_frac: float,
                      clay_frac: float, height_rms: float, corr_len: float,
                      inter_layer_corr: float, mean_total_depth: float,
                      top_layer_height: float, count: int, seed: int,
                      grooved: bool = False, rabus_compat: bool = False):
    """
    Yields a series of MaterialModel's containing a two layer dielectric with
    Gaussian correlated rough surfaces. The top layer contains a continuous
    gradient between top_moisture and bottom_moistures. The user can specify
    the correlation between the two layers. The dielectric properties are
    computed using the Hallikainen model.
    """
    nx = params.geom_params.size_x
    ny = params.geom_params.size_y
    nz = params.geom_params.size_z_material
    dtype = params.processing_params.datatype
    dx = params.geom_params.cell_size
    max_height = nz * dx
    total_depth_cells = mean_total_depth / dx
    top_layer_cells = top_layer_height / dx
    freq = params.inc_beam_params.frequency

    if mean_total_depth > max_height:
        raise ValueError('Requested mean soil depth is greater than the max '
                         'height of the simulator space.')

    if top_layer_cells > total_depth_cells:
        raise ValueError('Top layer of soil is thicker than entire soil model')

    # We need a small numba jit'ed function for filling in the material
    # arrays to be fast
    @numba.njit
    def fill_in_material(_eps_arr, _sigma_arr, _surf_height, _sm_top,
                         _sm_bot):
        _eps_cplx_bot = hallikainen_soil_model(_sm_bot, sand_frac, clay_frac,
                                               freq)
        _eps_real_bot = np.real(_eps_cplx_bot)
        _sigma_bot = np.abs(np.imag(_eps_cplx_bot)) * epsilon_0 * 2.0\
             * np.pi * freq
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if z <= _surf_height[0, y, x]:
                        # Bottom part of the soil has a constant moisture
                        _eps_arr[z, y, x] = _eps_real_bot
                        _sigma_arr[z, y, x] = _sigma_bot
                    elif z <= _surf_height[1, y, x]:
                        # Top part of the soil has a moisture gradient
                        # Linear interpolate soil moisture
                        # Equation shamelessly grabbed from https://en.wikipedia.org/wiki/Linear_interpolation
                        _sm0 = _sm_bot
                        _sm1 = _sm_top
                        _h0 = _surf_height[0, y, x]
                        _h1 = _surf_height[1, y, x]
                        _sm = ((_sm0 * (_h1 - z)) + (_sm1 * (z - _h0)))\
                            / (_h1 - _h0)

                        # Compute material properties with Hallikainen model
                        _eps_cplx = hallikainen_soil_model(_sm, sand_frac,
                                                           clay_frac, freq)
                        _eps_arr[z, y, x] = np.real(_eps_cplx)
                        _sigma_arr[z, y, x] = np.abs(np.imag(_eps_cplx))\
                            * epsilon_0 * 2.0 * np.pi * freq

    for i in range(count):
        surf_height = two_layer_gaussian_correlated_surface(
            nx, ny, dx, corr_len, corr_len, height_rms, inter_layer_corr,
            seed + i, rabus_compat=rabus_compat)
        if grooved:
            groove_profile = surf_height[:, 0, :]
            groove_profile = groove_profile[:, np.newaxis, :]
            surf_height = np.repeat(groove_profile, ny, axis=1)

        surf_height /= dx
        surf_height += total_depth_cells
        surf_height[0, :, :] -= top_layer_cells
        surf_height = np.rint(surf_height).astype(np.int32)

        for moisture in bottom_moistures:
            metadata = {
                'top_moisture': top_moisture,
                'bottom_moisture': moisture,
                'sand_frac': sand_frac,
                'clay_frac': clay_frac,
                'height_rms': height_rms,
                'mean_total_depth': mean_total_depth,
                'total_depth_cells': total_depth_cells,
                'top_layer_height': top_layer_height,
                'top_layer_cells': top_layer_cells,
                'inter_layer_corr': inter_layer_corr,
                'corr_len': corr_len
            }

            name = f'mv_{moisture:.2f}_num_{i}'

            # Create arrays of vacuum material
            eps = np.ones((nz, ny, nx), dtype=dtype)
            sigma = np.zeros((nz, ny, nx), dtype=dtype)
            fill_in_material(eps, sigma, surf_height, top_moisture, moisture)
            m = MaterialModel(eps, sigma, dx, name, metadata)
            yield m
