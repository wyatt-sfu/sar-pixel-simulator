from typing import Mapping

import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt


def plot_phasors(phasors: Mapping[str, npt.ArrayLike]):
    """
    Plot the phasors on every side of the simulation space and the incident
    beam phasors.
    """
    z0 = 376.730313668  # [Ohm]

    # Colorbar fiddle factors
    nz = phasors['ex_ph_f'].shape[0]
    ny = phasors['ex_ph_d'].shape[0]
    nx = phasors['ex_ph_f'].shape[1]
    im_ratio_1 = nz / ny
    im_ratio_2 = nz / nx
    im_ratio_3 = ny / nx
    magic = 0.047

    ###########################################################################
    # Plot incident phasors
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Phasors: Incident', fontsize=16)
    im_ex = ax[0, 0].imshow(np.real(phasors['ex_inc_phasor']), origin='lower')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')
    ax[0, 0].set_title('Re($E_x$)')
    fig.colorbar(im_ex, ax=ax[0, 0], fraction=magic * im_ratio_3)

    im_ey = ax[0, 1].imshow(np.real(phasors['ey_inc_phasor']), origin='lower')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Y')
    ax[0, 1].set_title('Re($E_y$)')
    fig.colorbar(im_ey, ax=ax[0, 1], fraction=magic * im_ratio_3)

    im_hx = ax[1, 0].imshow(np.real(phasors['hx_inc_phasor']) * z0,
                            origin='lower')
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')
    ax[1, 0].set_title('Re($H_x$)')
    fig.colorbar(im_hx, ax=ax[1, 0], fraction=magic * im_ratio_3)

    im_hz = ax[1, 1].imshow(np.real(phasors['hy_inc_phasor']) * z0,
                            origin='lower')
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Y')
    ax[1, 1].set_title('Re($H_y$)')
    fig.colorbar(im_hz, ax=ax[1, 1], fraction=magic * im_ratio_3)

    ###########################################################################
    # Plot down phasors
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Phasors: Down Face', fontsize=16)
    im_ex = ax[0, 0].imshow(np.real(phasors['ex_ph_d']), origin='lower')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')
    ax[0, 0].set_title('Re($E_x$)')
    fig.colorbar(im_ex, ax=ax[0, 0], fraction=magic * im_ratio_3)

    im_ey = ax[0, 1].imshow(np.real(phasors['ey_ph_d']), origin='lower')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Y')
    ax[0, 1].set_title('Re($E_y$)')
    fig.colorbar(im_ey, ax=ax[0, 1], fraction=magic * im_ratio_3)

    im_hx = ax[1, 0].imshow(np.real(phasors['hx_ph_d']) * z0, origin='lower')
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')
    ax[1, 0].set_title('Re($H_x$)')
    fig.colorbar(im_hx, ax=ax[1, 0], fraction=magic * im_ratio_3)

    im_hz = ax[1, 1].imshow(np.real(phasors['hy_ph_d']) * z0, origin='lower')
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Y')
    ax[1, 1].set_title('Re($H_y$)')
    fig.colorbar(im_hz, ax=ax[1, 1], fraction=magic * im_ratio_3)

    ###########################################################################
    # Plot up phasors
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Phasors: Up Face', fontsize=16)
    im_ex = ax[0, 0].imshow(np.real(phasors['ex_ph_u']), origin='lower')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')
    ax[0, 0].set_title('Re($E_x$)')
    fig.colorbar(im_ex, ax=ax[0, 0], fraction=magic * im_ratio_3)

    im_ey = ax[0, 1].imshow(np.real(phasors['ey_ph_u']), origin='lower')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Y')
    ax[0, 1].set_title('Re($E_y$)')
    fig.colorbar(im_ey, ax=ax[0, 1], fraction=magic * im_ratio_3)

    im_hx = ax[1, 0].imshow(np.real(phasors['hx_ph_u']) * z0, origin='lower')
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')
    ax[1, 0].set_title('Re($H_x$)')
    fig.colorbar(im_hx, ax=ax[1, 0], fraction=magic * im_ratio_3)

    im_hz = ax[1, 1].imshow(np.real(phasors['hy_ph_u']) * z0, origin='lower')
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Y')
    ax[1, 1].set_title('Re($H_y$)')
    fig.colorbar(im_hz, ax=ax[1, 1], fraction=magic * im_ratio_3)

    ###########################################################################
    # Plot left phasors
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Phasors: Left Face', fontsize=16)
    im_ex = ax[0, 0].imshow(np.real(phasors['ey_ph_l']), origin='lower')
    ax[0, 0].set_xlabel('Y')
    ax[0, 0].set_ylabel('Z')
    ax[0, 0].set_title('Re($E_y$)')
    fig.colorbar(im_ex, ax=ax[0, 0], fraction=magic * im_ratio_1)

    im_ey = ax[0, 1].imshow(np.real(phasors['ez_ph_l']), origin='lower')
    ax[0, 1].set_xlabel('Y')
    ax[0, 1].set_ylabel('Z')
    ax[0, 1].set_title('Re($E_z$)')
    fig.colorbar(im_ey, ax=ax[0, 1], fraction=magic * im_ratio_1)

    im_hx = ax[1, 0].imshow(np.real(phasors['hy_ph_l']) * z0, origin='lower')
    ax[1, 0].set_xlabel('Y')
    ax[1, 0].set_ylabel('Z')
    ax[1, 0].set_title('Re($H_y$)')
    fig.colorbar(im_hx, ax=ax[1, 0], fraction=magic * im_ratio_1)

    im_hz = ax[1, 1].imshow(np.real(phasors['hz_ph_l']) * z0, origin='lower')
    ax[1, 1].set_xlabel('Y')
    ax[1, 1].set_ylabel('Z')
    ax[1, 1].set_title('Re($H_z$)')
    fig.colorbar(im_hz, ax=ax[1, 1], fraction=magic * im_ratio_1)

    ###########################################################################
    # Plot right phasors
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Phasors: Right Face', fontsize=16)
    im_ex = ax[0, 0].imshow(np.real(phasors['ey_ph_r']), origin='lower')
    ax[0, 0].set_xlabel('Y')
    ax[0, 0].set_ylabel('Z')
    ax[0, 0].set_title('Re($E_y$)')
    fig.colorbar(im_ex, ax=ax[0, 0], fraction=magic * im_ratio_1)

    im_ey = ax[0, 1].imshow(np.real(phasors['ez_ph_r']), origin='lower')
    ax[0, 1].set_xlabel('Y')
    ax[0, 1].set_ylabel('Z')
    ax[0, 1].set_title('Re($E_z$)')
    fig.colorbar(im_ey, ax=ax[0, 1], fraction=magic * im_ratio_1)

    im_hx = ax[1, 0].imshow(np.real(phasors['hy_ph_r']) * z0, origin='lower')
    ax[1, 0].set_xlabel('Y')
    ax[1, 0].set_ylabel('Z')
    ax[1, 0].set_title('Re($H_y$)')
    fig.colorbar(im_hx, ax=ax[1, 0], fraction=magic * im_ratio_1)

    im_hz = ax[1, 1].imshow(np.real(phasors['hz_ph_r']) * z0, origin='lower')
    ax[1, 1].set_xlabel('Y')
    ax[1, 1].set_ylabel('Z')
    ax[1, 1].set_title('Re($H_z$)')
    fig.colorbar(im_hz, ax=ax[1, 1], fraction=magic * im_ratio_1)

    ###########################################################################
    # Plot front phasors
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Phasors: Front Face', fontsize=16)
    im_ex = ax[0, 0].imshow(np.real(phasors['ex_ph_f']), origin='lower')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Z')
    ax[0, 0].set_title('Re($E_x$)')
    fig.colorbar(im_ex, ax=ax[0, 0], fraction=magic * im_ratio_2)

    im_ey = ax[0, 1].imshow(np.real(phasors['ez_ph_f']), origin='lower')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Z')
    ax[0, 1].set_title('Re($E_z$)')
    fig.colorbar(im_ey, ax=ax[0, 1], fraction=magic * im_ratio_2)

    im_hx = ax[1, 0].imshow(np.real(phasors['hx_ph_f']) * z0, origin='lower')
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Z')
    ax[1, 0].set_title('Re($H_x$)')
    fig.colorbar(im_hx, ax=ax[1, 0], fraction=magic * im_ratio_2)

    im_hz = ax[1, 1].imshow(np.real(phasors['hz_ph_f']) * z0, origin='lower')
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Z')
    ax[1, 1].set_title('Re($H_z$)')
    fig.colorbar(im_hz, ax=ax[1, 1], fraction=magic * im_ratio_2)

    ###########################################################################
    # Plot back phasors
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Phasors: Back Face', fontsize=16)
    im_ex = ax[0, 0].imshow(np.real(phasors['ex_ph_b']), origin='lower')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Z')
    ax[0, 0].set_title('Re($E_x$)')
    fig.colorbar(im_ex, ax=ax[0, 0], fraction=magic * im_ratio_2)

    im_ey = ax[0, 1].imshow(np.real(phasors['ez_ph_b']), origin='lower')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Z')
    ax[0, 1].set_title('Re($E_z$)')
    fig.colorbar(im_ey, ax=ax[0, 1], fraction=magic * im_ratio_2)

    im_hx = ax[1, 0].imshow(np.real(phasors['hx_ph_b']) * z0, origin='lower')
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Z')
    ax[1, 0].set_title('Re($H_x$)')
    fig.colorbar(im_hx, ax=ax[1, 0], fraction=magic * im_ratio_2)

    im_hz = ax[1, 1].imshow(np.real(phasors['hz_ph_b']) * z0, origin='lower')
    ax[1, 1].set_xlabel('X')
    ax[1, 1].set_ylabel('Z')
    ax[1, 1].set_title('Re($H_z$)')
    fig.colorbar(im_hz, ax=ax[1, 1], fraction=magic * im_ratio_2)
