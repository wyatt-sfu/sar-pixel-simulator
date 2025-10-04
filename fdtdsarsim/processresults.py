from typing import Mapping
import glob
import os.path

from tqdm import tqdm
import h5py
import numpy as np
import numpy.typing as npt
import scipy.integrate

from fdtdsarsim.near2far import FarFieldTransform
from fdtdsarsim.utils.plotting.phasors import plot_phasors


def process_all_results(project_dir: str, debug_plots=False):
    """ Process all the results from a simulation run.
    """
    result_dir = os.path.join(project_dir, 'results')
    result_files = glob.glob(os.path.join(result_dir, '*.hdf5'))

    for result_file in tqdm(result_files, desc='Processing results'):
        process_results(result_file, debug_plots)


def process_results(filename: str, debug_plots=False):
    """ Process the results from the specified results file.

    The filename must specify an HDF5 file created by running a simulation.
    The results are placed back into the file specified.
    """
    # Constants used for near2far
    radius = 2.0e3
    thetas = np.linspace(-80.0, 80.0, num=161, endpoint=True)

    with h5py.File(filename, 'r+') as results:
        #######################################################################
        # Read metadata
        param_grp = results['parameters']
        phasor_grp = results['phasors']

        dx = param_grp.attrs['cell_size']
        size_z_material = param_grp.attrs['size_z_material']
        nx_phasor = phasor_grp.attrs['nx_phasor']
        ny_phasor = phasor_grp.attrs['ny_phasor']
        nz_phasor = phasor_grp.attrs['nz_phasor']
        #######################################################################

        #######################################################################
        # Compute far field and write into the HDF5 file
        if 'farfield' not in results:
            ff_grp = results.create_group('farfield')
        else:
            ff_grp = results['farfield']

        ff_attrs = ff_grp.attrs
        if 'farfields_computed' not in ff_attrs:
            phis = np.asarray([param_grp.attrs['phi_angle']])
            freq = param_grp.attrs['frequency']
            ff_transform = FarFieldTransform()
            farfields = ff_transform(results['phasors'], nx_phasor, ny_phasor,
                                     nz_phasor, dx, radius, thetas, phis, freq)
            ff_grp.create_dataset('e_theta', data=farfields['E_theta'])
            ff_grp.create_dataset('e_phi', data=farfields['E_phi'])
            ff_grp.create_dataset('h_theta', data=farfields['H_theta'])
            ff_grp.create_dataset('h_phi', data=farfields['H_phi'])

            ff_attrs['theta_angles'] = thetas
            ff_attrs['phi_angles'] = phis
            ff_attrs['radius'] = radius

            # Compute scattered power in horizontal and vertical polarizations
            out_pwr_far_h, out_pwr_far_v = farfield_power(
                farfields['E_theta'], farfields['E_phi'], farfields['H_theta'],
                farfields['H_phi'])
            ff_grp.create_dataset('out_pwr_far_h', data=out_pwr_far_h)
            ff_grp.create_dataset('out_pwr_far_v', data=out_pwr_far_v)

            ff_attrs['farfields_computed'] = True
        #######################################################################

        #######################################################################
        # Compute power using the Poynting vector
        if 'near_field_power' not in results:
            nf_pwr_grp = results.create_group('near_field_power')
        else:
            nf_pwr_grp = results['near_field_power']

        nf_pwr_attrs = nf_pwr_grp.attrs
        if 'powers_computed' not in nf_pwr_attrs:
            nf_pwr = near_field_power(results['phasors'], dx, size_z_material)
            for key, value in nf_pwr.items():
                nf_pwr_grp.attrs[key] = value
            nf_pwr_grp.attrs['powers_computed'] = True
        #######################################################################

        #######################################################################
        # Compute scattering coefficient
        if 'scattering_coefficient' not in results:
            scattering_grp = results.create_group('scattering_coefficient')
        else:
            scattering_grp = results['scattering_coefficient']

        scattering_attrs = scattering_grp.attrs
        if 'scattering_computed' not in scattering_attrs:
            # Get the total incident power
            input_power = nf_pwr_attrs['incident']

            # Normalize based on input power
            h_pwr = ff_grp['out_pwr_far_h'] / input_power
            v_pwr = ff_grp['out_pwr_far_v'] / input_power

            # Compute the scattering coefficient
            scattering_coeff_h = h_pwr * 4.0 * np.pi * (radius**2) \
                / (nx_phasor * ny_phasor * dx * dx)
            scattering_coeff_v = v_pwr * 4.0 * np.pi * (radius**2) \
                / (nx_phasor * ny_phasor * dx * dx)
            scattering_grp.create_dataset('scattering_coeff_h',
                                          data=scattering_coeff_h)
            scattering_grp.create_dataset('scattering_coeff_v',
                                          data=scattering_coeff_v)
            scattering_attrs['scattering_computed'] = True
        #######################################################################

        if debug_plots:
            plot_phasors(results['phasors'])


def near_field_power(phasors: Mapping[str, npt.ArrayLike], dx: float,
                     size_z_material: int):
    """
    Compute the near field power over the internal faces of the simulation
    space.

    Args:
        phasors: Near field phasors recoreded by the simulation
        dx: Spacing between phasor locations [m]
        size_z_material: Number of Z cells in the bottom section of the
            simulation space.

    Returns:
        Mapping[str, float]: Power in Watts for each of the specified areas.

    """
    power = {}

    # Power in incident beam
    E = np.stack([phasors['ex_inc_phasor'], phasors['ey_inc_phasor'],
                  np.zeros_like(phasors['ey_inc_phasor'])], axis=-1)
    H = np.stack([phasors['hx_inc_phasor'], phasors['hy_inc_phasor'],
                  np.zeros_like(phasors['hy_inc_phasor'])], axis=-1)
    norm_vec = np.asarray([0.0, 0.0, -1.0])
    inc_pwr_xy = np.dot(np.real(0.5 * np.cross(E, np.conj(H))), norm_vec)
    inc_pwr_y = scipy.integrate.simpson(inc_pwr_xy, dx=dx, axis=1)
    inc_pwr = scipy.integrate.simpson(inc_pwr_y, dx=dx, axis=0)
    power['incident'] = inc_pwr

    # Power in left side of simulation space
    E = np.stack([np.zeros_like(phasors['ey_ph_l']), phasors['ey_ph_l'],
                  phasors['ez_ph_l']], axis=-1)
    H = np.stack([np.zeros_like(phasors['hy_ph_l']), phasors['hy_ph_l'],
                  phasors['hz_ph_l']], axis=-1)
    norm_vec = np.asarray([-1.0, 0.0, 0.0])
    left_pwr_yz = np.dot(np.real(0.5 * np.cross(E, np.conj(H))), norm_vec)

    left_pwr_yz_bot = left_pwr_yz[0:size_z_material, :]
    left_pwr_z_bot = scipy.integrate.simpson(left_pwr_yz_bot, dx=dx, axis=1)
    left_pwr_bot = scipy.integrate.simpson(left_pwr_z_bot, dx=dx, axis=0)
    power['left_bottom'] = left_pwr_bot

    left_pwr_yz_top = left_pwr_yz[size_z_material:-1, :]
    left_pwr_z_top = scipy.integrate.simpson(left_pwr_yz_top, dx=dx, axis=1)
    left_pwr_top = scipy.integrate.simpson(left_pwr_z_top, dx=dx, axis=0)
    power['left_top'] = left_pwr_top

    # Power in right side of simulation space
    E = np.stack([np.zeros_like(phasors['ey_ph_r']), phasors['ey_ph_r'],
                  phasors['ez_ph_r']], axis=-1)
    H = np.stack([np.zeros_like(phasors['hy_ph_r']), phasors['hy_ph_r'],
                  phasors['hz_ph_r']], axis=-1)
    norm_vec = np.asarray([1.0, 0.0, 0.0])
    right_pwr_yz = np.dot(np.real(0.5 * np.cross(E, np.conj(H))), norm_vec)

    right_pwr_yz_bot = right_pwr_yz[0:size_z_material, :]
    right_pwr_z_bot = scipy.integrate.simpson(right_pwr_yz_bot, dx=dx, axis=1)
    right_pwr_bot = scipy.integrate.simpson(right_pwr_z_bot, dx=dx, axis=0)
    power['right_bottom'] = right_pwr_bot

    right_pwr_yz_top = right_pwr_yz[size_z_material:-1, :]
    right_pwr_z_top = scipy.integrate.simpson(right_pwr_yz_top, dx=dx, axis=1)
    right_pwr_top = scipy.integrate.simpson(right_pwr_z_top, dx=dx, axis=0)
    power['right_top'] = right_pwr_top

    # Power in front side of simulation space
    E = np.stack([phasors['ex_ph_f'], np.zeros_like(phasors['ex_ph_f']),
                  phasors['ez_ph_f']], axis=-1)
    H = np.stack([phasors['hx_ph_f'], np.zeros_like(phasors['hx_ph_f']),
                  phasors['hz_ph_f']], axis=-1)
    norm_vec = np.asarray([0.0, -1.0, 0.0])
    front_pwr_xz = np.dot(np.real(0.5 * np.cross(E, np.conj(H))), norm_vec)

    front_pwr_xz_bot = front_pwr_xz[0:size_z_material, :]
    front_pwr_z_bot = scipy.integrate.simpson(front_pwr_xz_bot, dx=dx, axis=1)
    front_pwr_bot = scipy.integrate.simpson(front_pwr_z_bot, dx=dx, axis=0)
    power['front_bottom'] = front_pwr_bot

    front_pwr_xz_top = front_pwr_xz[size_z_material:-1, :]
    front_pwr_z_top = scipy.integrate.simpson(front_pwr_xz_top, dx=dx, axis=1)
    front_pwr_top = scipy.integrate.simpson(front_pwr_z_top, dx=dx, axis=0)
    power['front_top'] = front_pwr_top

    # Power in back side of simulation space
    E = np.stack([phasors['ex_ph_b'], np.zeros_like(phasors['ex_ph_b']),
                  phasors['ez_ph_b']], axis=-1)
    H = np.stack([phasors['hx_ph_b'], np.zeros_like(phasors['hx_ph_b']),
                  phasors['hz_ph_b']], axis=-1)
    norm_vec = np.asarray([0.0, 1.0, 0.0])
    back_pwr_xz = np.dot(np.real(0.5 * np.cross(E, np.conj(H))), norm_vec)

    back_pwr_xz_bot = back_pwr_xz[0:size_z_material, :]
    back_pwr_z_bot = scipy.integrate.simpson(back_pwr_xz_bot, dx=dx, axis=1)
    back_pwr_bot = scipy.integrate.simpson(back_pwr_z_bot, dx=dx, axis=0)
    power['back_bottom'] = back_pwr_bot

    back_pwr_xz_top = back_pwr_xz[size_z_material:-1, :]
    back_pwr_z_top = scipy.integrate.simpson(back_pwr_xz_top, dx=dx, axis=1)
    back_pwr_top = scipy.integrate.simpson(back_pwr_z_top, dx=dx, axis=0)
    power['back_top'] = back_pwr_top

    # Power in down side of simulation space
    E = np.stack([phasors['ex_ph_d'], phasors['ey_ph_d'],
                  np.zeros_like(phasors['ex_ph_d'])], axis=-1)
    H = np.stack([phasors['hx_ph_d'], phasors['hy_ph_d'],
                  np.zeros_like(phasors['hx_ph_d'])], axis=-1)
    norm_vec = np.asarray([0.0, 0.0, -1.0])
    down_pwr_xy = np.dot(np.real(0.5 * np.cross(E, np.conj(H))), norm_vec)
    down_pwr_y = scipy.integrate.simpson(down_pwr_xy, dx=dx, axis=1)
    down_pwr = scipy.integrate.simpson(down_pwr_y, dx=dx, axis=0)
    power['down'] = down_pwr

    # Power in up side of simulation space
    E = np.stack([phasors['ex_ph_u'], phasors['ey_ph_u'],
                  np.zeros_like(phasors['ex_ph_u'])], axis=-1)
    H = np.stack([phasors['hx_ph_u'], phasors['hy_ph_u'],
                  np.zeros_like(phasors['hx_ph_u'])], axis=-1)
    norm_vec = np.asarray([0.0, 0.0, 1.0])
    up_pwr_xy = np.dot(np.real(0.5 * np.cross(E, np.conj(H))), norm_vec)
    up_pwr_y = scipy.integrate.simpson(up_pwr_xy, dx=dx, axis=1)
    up_pwr = scipy.integrate.simpson(up_pwr_y, dx=dx, axis=0)
    power['up'] = up_pwr

    # Sum up total in top halfspace and bottom halfspace
    power['top_total'] = up_pwr + left_pwr_top + right_pwr_top \
        + front_pwr_top + back_pwr_top
    power['bottom_total'] = down_pwr + left_pwr_bot + right_pwr_bot \
        + front_pwr_bot + back_pwr_bot

    return power


def farfield_power(e_theta: np.ndarray, e_phi: np.ndarray,
                   h_theta: np.ndarray, h_phi: np.ndarray):
    """
    Compute the power of the scattered fields in both horizontal and vertical
    polarizations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Horizontal, vertical) scattered power
    """
    out_pwr_h = 0.5 * np.real(-e_phi * np.conj(h_theta))
    out_pwr_v = 0.5 * np.real(e_theta * np.conj(h_phi))
    return out_pwr_h, out_pwr_v
