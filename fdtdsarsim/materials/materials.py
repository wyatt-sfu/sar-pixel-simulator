"""
The materials module defines any of the material model generators that come
"out-of-the-box" with this library. However, they can (and should) be used
as examples for how to build new material models for any future experiments.

At a high level, a material model is a generator which yields a MaterialModel
object. The model under test should take up as much space as possible in these
arrays because the required vacuum will be added on top.

Using generators is important because is allows the experiment runner to simply
loop through material models without knowing how models are stored or
generated. It is equally valid to save them to disk and use a generator to
read them, or to create them on the fly.
"""
import numpy as np
import numba

from fdtdsarsim.parameters import ParameterCollection
from .materialmodel import MaterialModel
from .roughsurface import gaussian_correlated_surface


def compute_material_arrays(params: ParameterCollection,
                            materials: MaterialModel):
    """
    Compute the derived material arrays from permittivity and conductivity
    arrays. This converts the real material values to the derived values
    required by the FDTD simulation.

    Args:
        params (ParameterCollection): Simulation parameters
        material_spec (MaterialModel): A model from a material generator

    Returns:
        Tuple(np.ndarray, np.ndarray): Ca and Cb arrays (Taflove Eqn 3.31a,b)
    """
    nx = params.geom_params.size_x
    ny = params.geom_params.size_y
    nz = params.geom_params.size_z
    nz_material = params.geom_params.size_z_material
    dx = params.geom_params.cell_size
    dt = params.geom_params.timestep
    dtype = params.processing_params.datatype

    eps0 = 8.8541878128e-12  # Vacuum permittivity

    if materials.relative_permittivity.shape != materials.conductivity.shape:
        raise RuntimeError('Permittivity and conductivity arrays have '
                           'different shapes')

    # Add vacuum to the top of the models
    vacuum_height = nz - nz_material
    vacuum_eps = np.ones((vacuum_height, ny, nx), dtype=dtype)
    vacuum_sig = np.zeros((vacuum_height, ny, nx), dtype=dtype)
    perm_full = np.concatenate((materials.relative_permittivity, vacuum_eps),
                               axis=0, dtype=dtype)
    perm_full *= eps0
    cond_full = np.concatenate((materials.conductivity, vacuum_sig), axis=0,
                               dtype=dtype)

    Ca = (1 - ((cond_full * dt) / (2.0 * perm_full))) \
        / (1 + ((cond_full * dt) / (2.0 * perm_full)))
    Cb = (dt / (perm_full * dx)) / (1 + ((cond_full * dt) / (2.0 * perm_full)))

    assert Ca.dtype == dtype
    assert Cb.dtype == dtype

    return (Ca, Cb)


def all_vacuum(params: ParameterCollection):
    """
    Yields a single MaterialModel containing pure vacuum. This generator can
    only be used when the geometric parameters have been pre-specified.
    """
    nx = params.geom_params.size_x
    ny = params.geom_params.size_y
    nz_material = params.geom_params.size_z_material
    dtype = params.processing_params.datatype

    epsilon_r = np.ones((nz_material, ny, nx), dtype=dtype)
    sigma = np.zeros((nz_material, ny, nx), dtype=dtype)

    m = MaterialModel(epsilon_r, sigma, params.geom_params.cell_size,
                      'vacuum')
    yield m


def flat_surface(params: ParameterCollection, epsilon_r: float,
                 conductivity: float, name='planar_surface'):
    """
    Yields a single MaterialModel containing a flat surface with the specified
    properties. This generator can only be used when the geometric parameters
    have been pre-specified.
    """
    nx = params.geom_params.size_x
    ny = params.geom_params.size_y
    nz_material = params.geom_params.size_z_material
    dtype = params.processing_params.datatype
    metadata = {
        'epsilon_r': epsilon_r,
        'conductivity': conductivity
    }

    eps = epsilon_r * np.ones((nz_material, ny, nx), dtype=dtype)
    sigma = conductivity * np.ones((nz_material, ny, nx), dtype=dtype)

    m = MaterialModel(eps, sigma, params.geom_params.cell_size, name, metadata)
    yield m


def rough_dielectric(params: ParameterCollection, epsilon_r: float,
                     conductivity: float, height_rms: float, mean_depth: float,
                     corr_len: float, count: int):
    """
    Yields a series of MaterialModel's containing a dielectric material with a
    Gaussian correlated rough surface.
    """
    nx = params.geom_params.size_x
    ny = params.geom_params.size_y
    nz = params.geom_params.size_z_material
    dtype = params.processing_params.datatype
    dx = params.geom_params.cell_size
    max_height = nz * dx
    mean_depth_cells = mean_depth / dx

    if mean_depth > max_height:
        raise ValueError('Requested mean soil depth is greater than the max '
                         'height of the simulator space.')
    metadata = {
        'epsilon_r': epsilon_r,
        'conductivity': conductivity,
        'height_rms': height_rms,
        'mean_depth': mean_depth,
        'mean_depth_cells': mean_depth_cells,
        'corr_len': corr_len
    }

    # We need a small numba jit'ed function for filling in the material
    # arrays to be fast
    @numba.njit
    def fill_in_material(_eps, _sigma, _surf_height):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if z <= _surf_height[y, x]:
                        _eps[z, y, x] = epsilon_r
                        _sigma[z, y, x] = conductivity

    # Create as many material arrays with random surfaces as requested
    for i in range(count):
        name = f'rough_surf_{i}'

        # Create arrays of vacuum material
        eps = np.ones((nz, ny, nx), dtype=dtype)
        sigma = np.zeros((nz, ny, nx), dtype=dtype)
        surf_height = gaussian_correlated_surface(nx, ny, dx, corr_len,
                                                  corr_len, height_rms)
        surf_height /= dx
        surf_height += mean_depth_cells
        fill_in_material(eps, sigma, surf_height)

        m = MaterialModel(eps, sigma, dx, name, metadata)
        yield m
