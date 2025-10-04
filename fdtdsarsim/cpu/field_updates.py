"""
This module contains the FDTD field update equations implemented on the CPU.

Equation numbers reference the book "Computational Electrodynamics: The
Finite-Difference Time-Domain Method" 3rd Ed. by Taflove and Hagness.

This is based off the Fortran example code in Section 7.9.2.
"""

import numpy as np
from numba import njit


#################################################
# E field update functions
#################################################

@njit(nogil=True)
def ex_update(ex: np.ndarray, hy: np.ndarray, hz: np.ndarray, ca: np.ndarray,
              cb: np.ndarray, start_z: int, end_z: int):
    """ Ex field component update (Eqn 3.29a, 7.106)
    """
    ny = ex.shape[1]
    nx = ex.shape[2]
    for z in range(start_z, end_z):
        for y in range(1, ny - 1):
            for x in range(0, nx - 1):
                scaled = ex[z, y, x] * ca[z, y, x]
                delta_hz = hz[z, y, x] - hz[z, y - 1, x]
                delta_hy = hy[z, y, x] - hy[z - 1, y, x]
                ex[z, y, x] = scaled + (cb[z, y, x] * (delta_hz - delta_hy))


@njit(nogil=True)
def ey_update(ey: np.ndarray, hx: np.ndarray, hz: np.ndarray, ca: np.ndarray,
              cb: np.ndarray, start_z: int, end_z: int):
    """ Ey field component update (Eqn 3.29b, 7.106)
    """
    ny = ey.shape[1]
    nx = ey.shape[2]
    for z in range(start_z, end_z):
        for y in range(0, ny - 1):
            for x in range(1, nx - 1):
                scaled = ey[z, y, x] * ca[z, y, x]
                delta_hx = hx[z, y, x] - hx[z - 1, y, x]
                delta_hz = hz[z, y, x] - hz[z, y, x - 1]
                ey[z, y, x] = scaled + (cb[z, y, x] * (delta_hx - delta_hz))


@njit(nogil=True)
def ez_update(ez: np.ndarray, hx: np.ndarray, hy: np.ndarray, ca: np.ndarray,
              cb: np.ndarray, start_z: int, end_z: int):
    """ Ez field component update (Eqn 3.29c, 7.106)
    """
    ny = ez.shape[1]
    nx = ez.shape[2]
    for z in range(start_z, end_z):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                scaled = ez[z, y, x] * ca[z, y, x]
                delta_hx = hx[z, y, x] - hx[z, y - 1, x]
                delta_hy = hy[z, y, x] - hy[z, y, x - 1]
                ez[z, y, x] = scaled + (cb[z, y, x] * (delta_hy - delta_hx))


@njit(nogil=True)
def cpml_e_down(ex: np.ndarray, ey: np.ndarray, hx: np.ndarray, hy: np.ndarray,
                psiExz_d: np.ndarray, psiEyz_d: np.ndarray, be_d: np.ndarray,
                ce_d: np.ndarray, cb: np.ndarray, dx: float):
    """ Update down CPML and correct Ey, Ex fields
    """
    npml = be_d.shape[0]
    ny = ex.shape[1]
    nx = ex.shape[2]
    for z in range(1, npml):
        for y in range(0, ny - 1):
            for x in range(0, nx - 1):
                psiEyz_d[z, y, x] = (be_d[z, y, x] * psiEyz_d[z, y, x]) + \
                    (ce_d[z, y, x] * (hx[z, y, x] - hx[z - 1, y, x]) / dx)
                psiExz_d[z, y, x] = (be_d[z, y, x] * psiExz_d[z, y, x]) + \
                    (ce_d[z, y, x] * (hy[z, y, x] - hy[z - 1, y, x]) / dx)
                ey[z, y, x] += cb[z, y, x] * dx * psiEyz_d[z, y, x]
                ex[z, y, x] -= cb[z, y, x] * dx * psiExz_d[z, y, x]


@njit(nogil=True)
def cpml_e_up(ex: np.ndarray, ey: np.ndarray, hx: np.ndarray, hy: np.ndarray,
              psiExz_u: np.ndarray, psiEyz_u: np.ndarray, be_u: np.ndarray,
              ce_u: np.ndarray, cb: np.ndarray, dx: float):
    """ Update up CPML and correct Ey, Ex fields
    """
    npml = be_u.shape[0]
    nz = ex.shape[0]
    ny = ex.shape[1]
    nx = ex.shape[2]
    for z in range(0, npml - 1):
        for y in range(0, ny - 1):
            for x in range(0, nx - 1):
                zf = nz - npml + z
                psiEyz_u[z, y, x] = be_u[z, y, x] * psiEyz_u[z, y, x] + \
                    (ce_u[z, y, x] * (hx[zf, y, x] - hx[zf - 1, y, x]) / dx)
                psiExz_u[z, y, x] = be_u[z, y, x] * psiExz_u[z, y, x] + \
                    (ce_u[z, y, x] * (hy[zf, y, x] - hy[zf - 1, y, x]) / dx)
                ey[zf, y, x] += cb[zf, y, x] * dx * psiEyz_u[z, y, x]
                ex[zf, y, x] -= cb[zf, y, x] * dx * psiExz_u[z, y, x]


@njit(nogil=True)
def cpml_e_left(ey: np.ndarray, ez: np.ndarray, hy: np.ndarray, hz: np.ndarray,
                psiEyx_l: np.ndarray, psiEzx_l: np.ndarray, be_l: np.ndarray,
                ce_l: np.ndarray, cb: np.ndarray, dx: float):
    """ Update left CPML and correct Ez, Ey fields
    """
    nz = ey.shape[0]
    ny = ey.shape[1]
    npml = be_l.shape[2]
    for z in range(0, nz - 1):
        for y in range(0, ny - 1):
            for x in range(1, npml):
                psiEzx_l[z, y, x] = (be_l[z, y, x] * psiEzx_l[z, y, x]) + \
                    (ce_l[z, y, x] * (hy[z, y, x] - hy[z, y, x - 1]) / dx)
                psiEyx_l[z, y, x] = (be_l[z, y, x] * psiEyx_l[z, y, x]) + \
                    (ce_l[z, y, x] * (hz[z, y, x] - hz[z, y, x - 1]) / dx)
                ez[z, y, x] += cb[z, y, x] * dx * psiEzx_l[z, y, x]
                ey[z, y, x] -= cb[z, y, x] * dx * psiEyx_l[z, y, x]


@njit(nogil=True)
def cpml_e_right(ey: np.ndarray, ez: np.ndarray, hy: np.ndarray,
                 hz: np.ndarray, psiEyx_r: np.ndarray, psiEzx_r: np.ndarray,
                 be_r: np.ndarray, ce_r: np.ndarray, cb: np.ndarray,
                 dx: np.floating):
    """ Update right CPML and correct Ez, Ey fields
    """
    nz = ey.shape[0]
    ny = ey.shape[1]
    nx = ey.shape[2]
    npml = be_r.shape[2]
    for z in range(0, nz - 1):
        for y in range(0, ny - 1):
            for x in range(0, npml - 1):
                xf = nx - npml + x
                psiEzx_r[z, y, x] = (be_r[z, y, x] * psiEzx_r[z, y, x]) + \
                    (ce_r[z, y, x] * (hy[z, y, xf] - hy[z, y, xf - 1]) / dx)
                psiEyx_r[z, y, x] = (be_r[z, y, x] * psiEyx_r[z, y, x]) + \
                    (ce_r[z, y, x] * (hz[z, y, xf] - hz[z, y, xf - 1]) / dx)
                ez[z, y, xf] += cb[z, y, xf] * dx * psiEzx_r[z, y, x]
                ey[z, y, xf] -= cb[z, y, xf] * dx * psiEyx_r[z, y, x]


@njit(nogil=True)
def cpml_e_front(ex: np.ndarray, ez: np.ndarray, hx: np.ndarray,
                 hz: np.ndarray, psiExy_f: np.ndarray, psiEzy_f: np.ndarray,
                 be_f: np.ndarray, ce_f: np.ndarray, cb: np.ndarray,
                 dx: float):
    """ Update front CPML and correct Ey, Ez fields
    """
    nz = ex.shape[0]
    npml = be_f.shape[1]
    nx = ex.shape[2]
    for z in range(0, nz - 1):
        for y in range(1, npml):
            for x in range(0, nx - 1):
                psiExy_f[z, y, x] = (be_f[z, y, x] * psiExy_f[z, y, x]) + \
                    (ce_f[z, y, x] * (hz[z, y, x] - hz[z, y - 1, x]) / dx)
                psiEzy_f[z, y, x] = (be_f[z, y, x] * psiEzy_f[z, y, x]) + \
                    (ce_f[z, y, x] * (hx[z, y, x] - hx[z, y - 1, x]) / dx)
                ex[z, y, x] += cb[z, y, x] * dx * psiExy_f[z, y, x]
                ez[z, y, x] -= cb[z, y, x] * dx * psiEzy_f[z, y, x]


@njit(nogil=True)
def cpml_e_back(ex: np.ndarray, ez: np.ndarray, hx: np.ndarray, hz: np.ndarray,
                psiExy_b: np.ndarray, psiEzy_b: np.ndarray, be_b: np.ndarray,
                ce_b: np.ndarray, cb: np.ndarray, dx: float):
    """ Update front CPML and correct Ey, Ez fields
    """
    nz = ex.shape[0]
    npml = be_b.shape[1]
    ny = ex.shape[1]
    nx = ex.shape[2]
    for z in range(0, nz - 1):
        for y in range(0, npml - 1):
            for x in range(0, nx - 1):
                yf = ny - npml + y
                psiExy_b[z, y, x] = (be_b[z, y, x] * psiExy_b[z, y, x]) + \
                    (ce_b[z, y, x] * (hz[z, yf, x] - hz[z, yf - 1, x]) / dx)
                psiEzy_b[z, y, x] = (be_b[z, y, x] * psiEzy_b[z, y, x]) + \
                    (ce_b[z, y, x] * (hx[z, yf, x] - hx[z, yf - 1, x]) / dx)
                ex[z, yf, x] += cb[z, yf, x] * dx * psiExy_b[z, y, x]
                ez[z, yf, x] -= cb[z, yf, x] * dx * psiEzy_b[z, y, x]


#################################################
# H field update functions
#################################################

@njit(nogil=True)
def hx_update(hx: np.ndarray, ey: np.ndarray, ez: np.ndarray, dtm0dx: float,
              start_z: int, end_z: int):
    """ Hx field component update (Eqn 3.30a, 7.108)
    """
    ny = hx.shape[1]
    nx = hx.shape[2]
    for z in range(start_z, end_z):
        for y in range(0, ny - 1):
            for x in range(1, nx - 1):
                delta_ez = ez[z, y + 1, x] - ez[z, y, x]
                delta_ey = ey[z + 1, y, x] - ey[z, y, x]
                hx[z, y, x] -= dtm0dx * (delta_ez - delta_ey)


@njit(nogil=True)
def hy_update(hy: np.ndarray, ex: np.ndarray, ez: np.ndarray, dtm0dx: float,
              start_z: int, end_z: int):
    """ Hx field component update (Eqn 3.30b, 7.108)
    """
    ny = hy.shape[1]
    nx = hy.shape[2]
    for z in range(start_z, end_z):
        for y in range(1, ny - 1):
            for x in range(0, nx - 1):
                delta_ex = ex[z + 1, y, x] - ex[z, y, x]
                delta_ez = ez[z, y, x + 1] - ez[z, y, x]
                hy[z, y, x] -= dtm0dx * (delta_ex - delta_ez)


@njit(nogil=True)
def hz_update(hz: np.ndarray, ex: np.ndarray, ey: np.ndarray, dtm0dx: float,
              start_z: int, end_z: int):
    """ Hx field component update (Eqn 3.30c, 7.108)
    """
    ny = hz.shape[1]
    nx = hz.shape[2]
    for z in range(start_z, end_z):
        for y in range(0, ny - 1):
            for x in range(0, nx - 1):
                delta_ex = ex[z, y + 1, x] - ex[z, y, x]
                delta_ey = ey[z, y, x + 1] - ey[z, y, x]
                hz[z, y, x] -= dtm0dx * (delta_ey - delta_ex)


@njit(nogil=True)
def cpml_h_down(ex: np.ndarray, ey: np.ndarray, hx: np.ndarray, hy: np.ndarray,
                psiHxz_d: np.ndarray, psiHyz_d: np.ndarray, bh_d: np.ndarray,
                ch_d: np.ndarray, dtm0: float, dx: float):
    """ Update down CPML and correct Hy, Hx fields
    """
    npml = bh_d.shape[0]
    ny = ex.shape[1]
    nx = ex.shape[2]
    for z in range(0, npml - 1):
        for y in range(0, ny - 1):
            for x in range(0, nx - 1):
                psiHyz_d[z, y, x] = (bh_d[z, y, x] * psiHyz_d[z, y, x]) + \
                    (ch_d[z, y, x] * (ex[z + 1, y, x] - ex[z, y, x]) / dx)
                psiHxz_d[z, y, x] = (bh_d[z, y, x] * psiHxz_d[z, y, x]) + \
                    (ch_d[z, y, x] * (ey[z + 1, y, x] - ey[z, y, x]) / dx)
                hy[z, y, x] -= dtm0 * psiHyz_d[z, y, x]
                hx[z, y, x] += dtm0 * psiHxz_d[z, y, x]


@njit(nogil=True)
def cpml_h_up(ex: np.ndarray, ey: np.ndarray, hx: np.ndarray, hy: np.ndarray,
              psiHxz_u: np.ndarray, psiHyz_u: np.ndarray, bh_u: np.ndarray,
              ch_u: np.ndarray, dtm0: float, dx: float):
    """ Update up CPML and correct Hy, Hx fields
    """
    npml = bh_u.shape[0]
    nz = ex.shape[0]
    ny = ex.shape[1]
    nx = ex.shape[2]
    for z in range(0, npml - 1):
        for y in range(0, ny - 1):
            for x in range(0, nx - 1):
                zf = nz - npml + z
                psiHyz_u[z, y, x] = (bh_u[z, y, x] * psiHyz_u[z, y, x]) + \
                    (ch_u[z, y, x] * (ex[zf + 1, y, x] - ex[zf, y, x]) / dx)
                psiHxz_u[z, y, x] = (bh_u[z, y, x] * psiHxz_u[z, y, x]) + \
                    (ch_u[z, y, x] * (ey[zf + 1, y, x] - ey[zf, y, x]) / dx)
                hy[zf, y, x] -= dtm0 * psiHyz_u[z, y, x]
                hx[zf, y, x] += dtm0 * psiHxz_u[z, y, x]


@njit(nogil=True)
def cpml_h_left(ey: np.ndarray, ez: np.ndarray, hy: np.ndarray, hz: np.ndarray,
                psiHyx_l: np.ndarray, psiHzx_l: np.ndarray, bh_l: np.ndarray,
                ch_l: np.ndarray, dtm0: float, dx: float):
    """ Update left CPML and correct Hz, Hy fields
    """
    nz = ey.shape[0]
    ny = ey.shape[1]
    npml = bh_l.shape[2]
    for z in range(0, nz - 1):
        for y in range(0, ny - 1):
            for x in range(0, npml - 1):
                psiHzx_l[z, y, x] = (bh_l[z, y, x] * psiHzx_l[z, y, x]) + \
                    (ch_l[z, y, x] * (ey[z, y, x + 1] - ey[z, y, x]) / dx)
                psiHyx_l[z, y, x] = (bh_l[z, y, x] * psiHyx_l[z, y, x]) + \
                    (ch_l[z, y, x] * (ez[z, y, x + 1] - ez[z, y, x]) / dx)
                hz[z, y, x] -= dtm0 * psiHzx_l[z, y, x]
                hy[z, y, x] += dtm0 * psiHyx_l[z, y, x]


@njit(nogil=True)
def cpml_h_right(ey: np.ndarray, ez: np.ndarray, hy: np.ndarray,
                 hz: np.ndarray, psiHyx_r: np.ndarray, psiHzx_r: np.ndarray,
                 bh_r: np.ndarray, ch_r: np.ndarray, dtm0: float, dx: float):
    """ Update right CPML and correct Hz, Hy fields
    """
    nz = ey.shape[0]
    ny = ey.shape[1]
    nx = ey.shape[2]
    npml = bh_r.shape[2]
    for z in range(0, nz - 1):
        for y in range(0, ny - 1):
            for x in range(0, npml - 1):
                xf = nx - npml + x
                psiHzx_r[z, y, x] = (bh_r[z, y, x] * psiHzx_r[z, y, x]) + \
                    (ch_r[z, y, x] * (ey[z, y, xf + 1] - ey[z, y, xf]) / dx)
                psiHyx_r[z, y, x] = (bh_r[z, y, x] * psiHyx_r[z, y, x]) + \
                    (ch_r[z, y, x] * (ez[z, y, xf + 1] - ez[z, y, xf]) / dx)
                hz[z, y, xf] -= dtm0 * psiHzx_r[z, y, x]
                hy[z, y, xf] += dtm0 * psiHyx_r[z, y, x]


@njit(nogil=True)
def cpml_h_front(ex: np.ndarray, ez: np.ndarray, hx: np.ndarray,
                 hz: np.ndarray, psiHxy_f: np.ndarray, psiHzy_f: np.ndarray,
                 bh_f: np.ndarray, ch_f: np.ndarray, dtm0: float, dx: float):
    """ Update front CPML and correct Hy, Hz fields
    """
    nz = ex.shape[0]
    npml = bh_f.shape[1]
    nx = ex.shape[2]
    for z in range(0, nz - 1):
        for y in range(0, npml - 1):
            for x in range(0, nx - 1):
                psiHxy_f[z, y, x] = (bh_f[z, y, x] * psiHxy_f[z, y, x]) + \
                    (ch_f[z, y, x] * (ez[z, y + 1, x] - ez[z, y, x]) / dx)
                psiHzy_f[z, y, x] = (bh_f[z, y, x] * psiHzy_f[z, y, x]) + \
                    (ch_f[z, y, x] * (ex[z, y + 1, x] - ex[z, y, x]) / dx)
                hx[z, y, x] -= dtm0 * psiHxy_f[z, y, x]
                hz[z, y, x] += dtm0 * psiHzy_f[z, y, x]


@njit(nogil=True)
def cpml_h_back(ex: np.ndarray, ez: np.ndarray, hx: np.ndarray, hz: np.ndarray,
                psiHxy_b: np.ndarray, psiHzy_b: np.ndarray, bh_b: np.ndarray,
                ch_b: np.ndarray, dtm0: float, dx: float):
    """ Update front CPML and correct Hy, Hz fields
    """
    nz = ex.shape[0]
    ny = ex.shape[1]
    npml = bh_b.shape[1]
    nx = ex.shape[2]
    for z in range(0, nz - 1):
        for y in range(0, npml - 1):
            for x in range(0, nx - 1):
                yf = ny - npml + y
                psiHxy_b[z, y, x] = (bh_b[z, y, x] * psiHxy_b[z, y, x]) + \
                    (ch_b[z, y, x] * (ez[z, yf + 1, x] - ez[z, yf, x]) / dx)
                psiHzy_b[z, y, x] = (bh_b[z, y, x] * psiHzy_b[z, y, x]) + \
                    (ch_b[z, y, x] * (ex[z, yf + 1, x] - ex[z, yf, x]) / dx)
                hx[z, yf, x] -= dtm0 * psiHxy_b[z, y, x]
                hz[z, yf, x] += dtm0 * psiHzy_b[z, y, x]


#################################################
# Phasor computation functions
#################################################

@njit(nogil=True)
def phasors_x_plane(ey: np.ndarray, ez: np.ndarray, hy: np.ndarray,
                    hz: np.ndarray, ey_ph: np.ndarray, ez_ph: np.ndarray,
                    hy_ph: np.ndarray, hz_ph: np.ndarray, omega: float,
                    phasor_x: int, t: int, dt: float, npml: int):
    """
    Compute phasors along the specified X plane
    """
    nz_ph = ey_ph.shape[0]
    ny_ph = ey_ph.shape[1]
    for z in range(1, nz_ph):
        for y in range(1, ny_ph):
            ey_ph[z, y] += 0.5 * (ey[npml + z, npml + y, phasor_x]
                                  + ey[npml + z, npml + y - 1, phasor_x]) \
                * np.exp(-1j * omega * t * dt)
            ez_ph[z, y] += 0.5 * (ez[npml + z, npml + y, phasor_x]
                                  + ez[npml + z - 1, npml + y, phasor_x]) \
                * np.exp(-1j * omega * t * dt)
            hy_ph[z, y] += 0.25 * (hy[npml + z, npml + y, phasor_x]
                                   + hy[npml + z, npml + y, phasor_x - 1]
                                   + hy[npml + z - 1, npml + y, phasor_x]
                                   + hy[npml + z - 1, npml + y, phasor_x - 1]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)
            hz_ph[z, y] += 0.25 * (hz[npml + z, npml + y, phasor_x]
                                   + hz[npml + z, npml + y, phasor_x - 1]
                                   + hz[npml + z, npml + y - 1, phasor_x]
                                   + hz[npml + z, npml + y - 1, phasor_x - 1]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)


@njit(nogil=True)
def phasors_y_plane(ex: np.ndarray, ez: np.ndarray, hx: np.ndarray,
                    hz: np.ndarray, ex_ph: np.ndarray, ez_ph: np.ndarray,
                    hx_ph: np.ndarray, hz_ph: np.ndarray, omega: float,
                    phasor_y: int, t: int, dt: float, npml: int):
    """
    Compute phasors along the specified Y plane
    """
    nz_ph = ex_ph.shape[0]
    nx_ph = ex_ph.shape[1]

    for z in range(1, nz_ph):
        for x in range(1, nx_ph):
            ex_ph[z, x] += 0.5 * (ex[npml + z, phasor_y, npml + x]
                                  + ex[npml + z, phasor_y, npml + x - 1]) \
                * np.exp(-1j * omega * t * dt)
            ez_ph[z, x] += 0.5 * (ez[npml + z, phasor_y, npml + x]
                                  + ez[npml + z - 1, phasor_y, npml + x]) \
                * np.exp(-1j * omega * t * dt)
            hx_ph[z, x] += 0.25 * (hx[npml + z, phasor_y, npml + x]
                                   + hx[npml + z, phasor_y - 1, npml + x]
                                   + hx[npml + z - 1, phasor_y, npml + x]
                                   + hx[npml + z - 1, phasor_y - 1, npml + x]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)
            hz_ph[z, x] += 0.25 * (hz[npml + z, phasor_y, npml + x]
                                   + hz[npml + z, phasor_y - 1, npml + x]
                                   + hz[npml + z, phasor_y, npml + x - 1]
                                   + hz[npml + z, phasor_y - 1, npml + x - 1]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)


@njit(nogil=True)
def phasors_z_plane(ex: np.ndarray, ey: np.ndarray, hx: np.ndarray,
                    hy: np.ndarray, ex_ph: np.ndarray, ey_ph: np.ndarray,
                    hx_ph: np.ndarray, hy_ph: np.ndarray, omega: float,
                    phasor_z: int, t: int, dt: float, npml: int):
    """
    Compute phasors along the specified Z plane
    """
    ny_ph = ex_ph.shape[0]
    nx_ph = ex_ph.shape[1]
    for y in range(1, ny_ph):
        for x in range(1, nx_ph):
            ex_ph[y, x] += 0.5 * (ex[phasor_z, npml + y, npml + x]
                                  + ex[phasor_z, npml + y, npml + x - 1]) \
                * np.exp(-1j * omega * t * dt)
            ey_ph[y, x] += 0.5 * (ey[phasor_z, npml + y, npml + x]
                                  + ey[phasor_z, npml + y - 1, npml + x]) \
                * np.exp(-1j * omega * t * dt)
            hx_ph[y, x] += 0.25 * (hx[phasor_z, npml + y, npml + x]
                                   + hx[phasor_z, npml + y - 1, npml + x]
                                   + hx[phasor_z - 1, npml + y, npml + x]
                                   + hx[phasor_z - 1, npml + y - 1, npml + x]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)
            hy_ph[y, x] += 0.25 * (hy[phasor_z, npml + y, npml + x]
                                   + hy[phasor_z, npml + y, npml + x - 1]
                                   + hy[phasor_z - 1, npml + y, npml + x]
                                   + hy[phasor_z - 1, npml + y, npml + x - 1]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)


@njit(nogil=True)
def phasors_incident(ex: np.ndarray, ey: np.ndarray, hx: np.ndarray,
                     hy: np.ndarray, ex_ph: np.ndarray, ey_ph: np.ndarray,
                     hx_ph: np.ndarray, hy_ph: np.ndarray, omega: float,
                     phasor_z: int, t: int, dt: float, npml: int):
    """
    Compute incident phasors along the specified Z plane
    """
    ny_ph = ex_ph.shape[0]
    nx_ph = ex_ph.shape[1]
    for y in range(1, ny_ph):
        for x in range(1, nx_ph):
            ex_ph[y, x] += 0.5 * (ex[phasor_z, npml + y, npml + x]
                                  + ex[phasor_z, npml + y, npml + x - 1]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)
            ey_ph[y, x] += 0.5 * (ey[phasor_z, npml + y, npml + x]
                                  + ey[phasor_z, npml + y - 1, npml + x]) \
                * np.exp(-1j * omega * (t - 0.5) * dt)
            hx_ph[y, x] += 0.25 * (hx[phasor_z, npml + y, npml + x]
                                   + hx[phasor_z, npml + y - 1, npml + x]
                                   + hx[phasor_z - 1, npml + y, npml + x]
                                   + hx[phasor_z - 1, npml + y - 1, npml + x]) \
                * np.exp(-1j * omega * t * dt)
            hy_ph[y, x] += 0.25 * (hy[phasor_z, npml + y, npml + x]
                                   + hy[phasor_z, npml + y, npml + x - 1]
                                   + hy[phasor_z - 1, npml + y, npml + x]
                                   + hy[phasor_z - 1, npml + y, npml + x - 1]) \
                * np.exp(-1j * omega * t * dt)
