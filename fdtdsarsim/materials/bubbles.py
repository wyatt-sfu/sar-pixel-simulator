"""
This module can be used to add small bubbles to a soil model.
"""

import math

import numpy as np
import numba

from .materialmodel import MaterialModel


def material_volume(rel_permittivity, conductivity):
    """
    Returns the material volume as a number of FDTD cells.
    """
    vol = np.sum(np.logical_or(rel_permittivity != 1.0, conductivity != 0.0))
    return vol


def add_bubbles(
    mat_model: MaterialModel,
    surf_height_cells: np.ndarray,
    bubble_frac: float,
    bubble_radius: float,
    bubble_eps_r: float,
    bubble_cond: float,
    seed: int,
):
    """
    Add small bubbles to the provided material model.

    Args:
        mat_model (MaterialModel): Material model to add bubbles to
        surf_height_cells (np.ndarray): Surface height in cells.
            Size is [ny x nx].
        bubble_frac (float): Fraction of the soil volume that should be filled
            with bubbles (0-1)
        bubble_radius (float): [m] The bubble radius
        bubble_eps_r (float): Permittivity of the bubble
        bubble_cond (float): Conductivity of the bubble
        seed (int): Seed for random number generator

    Returns:
        MaterialModel: Material model containing bubbles
    """
    rng = np.random.default_rng(seed=seed)

    # Figure out the volume of soil and the target volume of the soil
    # after we add bubbles
    orig_vol = material_volume(mat_model.relative_permittivity, mat_model.conductivity)
    soil_vol = orig_vol
    target_vol = int(round(soil_vol * (1.0 - bubble_frac)))

    # Material model size
    nz = mat_model.conductivity.shape[0]
    ny = mat_model.conductivity.shape[1]
    nx = mat_model.conductivity.shape[2]

    # Convert bubble radius to cells
    radius_cells = bubble_radius / mat_model.cell_size
    radius_cells_int = int(math.ceil(radius_cells))

    @numba.njit
    def _fill_with_bubbles(_eps_r, _cond, _rng):
        _soil_vol = orig_vol
        while _soil_vol > target_vol:
            # Randomly generate the x,y,z coordinate of a bubble
            z_loc = _rng.uniform(low=radius_cells_int, high=nz - radius_cells_int - 1)
            y_loc = _rng.uniform(low=radius_cells_int, high=ny - radius_cells_int - 1)
            x_loc = _rng.uniform(low=radius_cells_int, high=nx - radius_cells_int - 1)

            if (
                np.rint(z_loc + radius_cells_int)
                > surf_height_cells[int(np.rint(y_loc)), int(np.rint(x_loc))]
            ):
                # Bubble would go above the surface
                continue

            for z_off in range(-radius_cells_int, radius_cells_int + 1):
                z_sqr = z_off**2
                for y_off in range(-radius_cells_int, radius_cells_int + 1):
                    y_sqr = y_off**2
                    for x_off in range(-radius_cells_int, radius_cells_int + 1):
                        _r = z_sqr + y_sqr + x_off**2
                        if _r < radius_cells**2:
                            z_bub = int(np.rint(z_loc + z_off))
                            y_bub = int(np.rint(y_loc + y_off))
                            x_bub = int(np.rint(x_loc + x_off))
                            _eps_r[z_bub, y_bub, x_bub] = bubble_eps_r
                            _cond[z_bub, y_bub, x_bub] = bubble_cond
                            _soil_vol -= 1

    # Create the output material model and update the metadata
    out = mat_model
    out.metadata["bubble_frac"] = bubble_frac
    out.metadata["bubble_radius"] = bubble_radius
    out.metadata["bubble_radius_cells"] = radius_cells
    out.metadata["bubble_eps_r"] = bubble_eps_r
    out.metadata["bubble_cond"] = bubble_cond

    # Call the jit'd function to add bubbles
    _fill_with_bubbles(out.relative_permittivity, out.conductivity, rng)

    return out


def add_bubbles_randsize(
    mat_model: MaterialModel,
    surf_height_cells: np.ndarray,
    bubble_frac: float,
    bubble_radius_min: float,
    bubble_radius_max: float,
    bubble_eps_r: float,
    bubble_cond: float,
    seed: int,
):
    """
    Add small bubbles to the provided material model with a uniform random
    distribution of bubble sizes.

    Args:
        mat_model (MaterialModel): Material model to add bubbles to
        surf_height_cells (np.ndarray): Surface height in cells.
            Size is [ny x nx].
        bubble_frac (float): Fraction of the soil volume that should be filled
            with bubbles (0-1)
        bubble_radius_min (float): [m] The minimum bubble radius
        bubble_radius_max (float): [m] The maximum bubble radius
        bubble_eps_r (float): Permittivity of the bubble
        bubble_cond (float): Conductivity of the bubble
        seed (int): Seed for random number generator

    Returns:
        MaterialModel: Material model containing bubbles
    """
    rng = np.random.default_rng(seed=seed)

    # Figure out the volume of soil and the target volume of the soil
    # after we add bubbles
    orig_vol = material_volume(mat_model.relative_permittivity, mat_model.conductivity)
    soil_vol = orig_vol
    target_vol = int(round(soil_vol * (1.0 - bubble_frac)))

    # Material model size
    nz = mat_model.conductivity.shape[0]
    ny = mat_model.conductivity.shape[1]
    nx = mat_model.conductivity.shape[2]

    # Convert bubble radius to cells
    cell_size = mat_model.cell_size

    @numba.njit
    def _fill_with_bubbles(_eps_r, _cond, _rng):
        _soil_vol = orig_vol
        while _soil_vol > target_vol:
            # Randomly generate the radius of a bubble
            bubble_radius = _rng.uniform(low=bubble_radius_min, high=bubble_radius_max)
            radius_cells = bubble_radius / cell_size
            radius_cells_int = np.rint(radius_cells)

            # Randomly generate the x,y,z coordinate of a bubble
            z_loc = _rng.uniform(low=radius_cells_int, high=nz - radius_cells_int - 1)
            y_loc = _rng.uniform(low=radius_cells_int, high=ny - radius_cells_int - 1)
            x_loc = _rng.uniform(low=radius_cells_int, high=nx - radius_cells_int - 1)

            if (
                np.rint(z_loc + radius_cells_int)
                > surf_height_cells[int(np.rint(y_loc)), int(np.rint(x_loc))]
            ):
                # Bubble would go above the surface
                continue

            for z_off in range(-radius_cells_int, radius_cells_int + 1):
                z_sqr = z_off**2
                for y_off in range(-radius_cells_int, radius_cells_int + 1):
                    y_sqr = y_off**2
                    for x_off in range(-radius_cells_int, radius_cells_int + 1):
                        _r = z_sqr + y_sqr + x_off**2
                        if _r < radius_cells**2:
                            z_bub = int(np.rint(z_loc + z_off))
                            y_bub = int(np.rint(y_loc + y_off))
                            x_bub = int(np.rint(x_loc + x_off))
                            _eps_r[z_bub, y_bub, x_bub] = bubble_eps_r
                            _cond[z_bub, y_bub, x_bub] = bubble_cond
                            _soil_vol -= 1

    # Create the output material model and update the metadata
    out = mat_model
    out.metadata["bubble_frac"] = bubble_frac
    out.metadata["bubble_radius_min"] = bubble_radius_min
    out.metadata["bubble_radius_max"] = bubble_radius_max
    out.metadata["bubble_eps_r"] = bubble_eps_r
    out.metadata["bubble_cond"] = bubble_cond

    # Call the jit'd function to add bubbles
    _fill_with_bubbles(out.relative_permittivity, out.conductivity, rng)

    return out
