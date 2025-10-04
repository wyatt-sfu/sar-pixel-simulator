"""
The incidentbeam module is used for generating the incident beam which is
injected into the FDTD simulation space.
"""
import os.path
import time

import scipy.constants
import numpy as np
import numba
from tqdm import trange
import h5py
import matplotlib.pyplot as plt

import fdtdsarsim.utils.log as log
from .parameters import ParameterCollection, inc_filename
from .utils.plotting.incident import (
    plot_pulse,
    incident_excitation_movie,
    incident_beam_movie
)
from .cpu.incident_sim import IncidentSimCpu


class IncidentBeam:
    logger = log.getLog(__name__)

    def __init__(self, params: ParameterCollection):
        """ This class holds an incident beam.

        Args:
            params (ParameterCollection): Collection of all simulation
                parameters
        """
        self.params = params
        self.inc_beam_params = params.inc_beam_params
        self.geom_params = params.geom_params
        self.processing_params = params.processing_params
        self.display_params = params.display_params

    def setup(self):
        """
        Sets up the incident beam for the simulation. This will generate the
        incident beam files if they do not currently exist.

        Note: generating the incident beam files can take a long time.
        """
        IncidentBeam.logger.info('Starting incident beam setup')
        t0 = time.monotonic()

        if not self.inc_beam_params.generate_new and self.inc_beam_exists():
            IncidentBeam.logger.info('Loading existing incident beam files')
            self.load_existing()
        else:
            IncidentBeam.logger.info('Generating new incident beam')
            self.create_pulse()
            pulse_ex, pulse_ey = self.create_inc_excitation()
            inc_sim = IncidentSimCpu(self.params)
            inc_sim.run(pulse_ex, pulse_ey)
            self.load_existing()

        t1 = time.monotonic()
        IncidentBeam.logger.info(
            f'Incident beam setup complete ({t1 - t0:.3f}s)')

        if self.params.display_params.inc_results:
            incident_beam_movie(inc_filename(self.params),
                                self.display_params.display_limit,
                                self.display_params.display_step)

    def load_existing(self):
        """ Reads in existing incident beam files.

        This function assumes that self.inc_beam_exists() has returned True.
        """
        # Read the data file
        filename = inc_filename(self.params)
        with h5py.File(filename, 'r') as f:
            self.check_array_sizes(f['ex'], f['ey'], f['hx'], f['hy'])
            self.check_metadata(f.attrs)

    def check_array_sizes(self, inc_ex: np.ndarray, inc_ey: np.ndarray,
                          inc_hx: np.ndarray, inc_hy: np.ndarray):
        """
        Checks the size of each component array against the shape defined in
        the geom_params definition. Raises a value error if a shape is
        incorrect.
        """
        correct_size = (self.geom_params.size_t_inc, self.geom_params.size_y,
                        self.geom_params.size_x)
        inc_fields = [inc_ex, inc_ey, inc_hx, inc_hy]
        for field in inc_fields:
            if field.shape != correct_size:
                raise ValueError(
                    f'A component of the incident beam has the wrong shape. '
                    f'Got {field.shape}, expected {correct_size}.')

            if field.dtype != self.processing_params.datatype:
                raise ValueError(
                    'Incident beam has incorrect datatype. Please delete the '
                    'incident beam file and re-run the simulation.')

        IncidentBeam.logger.info('Incident beam array sizes are valid')

    def check_metadata(self, meta):
        """
        Checks the metadata saved in the incident beam HDF5 file against the
        current parameters. An exception is thrown if the metadata is
        different.
        """
        proc_params = self.params.processing_params
        inc_params = self.params.inc_beam_params
        geom_params = self.params.geom_params

        if meta['datatype'] != proc_params.datatype:
            raise ValueError('Incident beam metadata inconsistency (datatype)')

        if meta['frequency'] != inc_params.frequency:
            raise ValueError('Incident beam metadata inconsistency '
                             '(frequency)')
        if meta['polarization'] != inc_params.polarization:
            raise ValueError('Incident beam metadata inconsistency '
                             '(polarization)')
        if meta['theta_angle'] != inc_params.theta_angle:
            raise ValueError('Incident beam metadata inconsistency '
                             '(theta_angle)')
        if meta['phi_angle'] != inc_params.phi_angle:
            raise ValueError('Incident beam metadata inconsistency '
                             '(phi_angle)')
        if meta['duration'] != inc_params.duration:
            raise ValueError('Incident beam metadata inconsistency '
                             '(duration)')
        if meta['illumination_width'] != inc_params.illumination_width:
            raise ValueError('Incident beam metadata inconsistency '
                             '(illumination_width)')

        if meta['cell_size'] != geom_params.cell_size:
            raise ValueError('Incident beam metadata inconsistency '
                             '(cell_size)')
        if meta['timestep'] != geom_params.timestep:
            raise ValueError('Incident beam metadata inconsistency '
                             '(timestep)')
        if meta['size_t_inc'] != geom_params.size_t_inc:
            raise ValueError('Incident beam metadata inconsistency '
                             '(duration)')
        if meta['size_t_pulse'] != geom_params.size_t_pulse:
            raise ValueError('Incident beam metadata inconsistency '
                             '(size_t_pulse)')
        if meta['size_x'] != geom_params.size_x:
            raise ValueError('Incident beam metadata inconsistency '
                             '(size_x)')
        if meta['size_y'] != geom_params.size_y:
            raise ValueError('Incident beam metadata inconsistency '
                             '(size_y)')
        if meta['size_pml'] != geom_params.size_pml:
            raise ValueError('Incident beam metadata inconsistency '
                             '(size_pml)')
        if meta['courant'] != geom_params.courant:
            raise ValueError('Incident beam metadata inconsistency '
                             '(courant)')

        if meta['incident_complete'] != True:  # noqa: E712
            raise ValueError('Incident beam simulation was not completed')

        IncidentBeam.logger.info('Incident beam metadata is valid')

    def inc_beam_exists(self):
        """
        Returns true if the incident beam file exists and all metadata matches.
        """
        name = inc_filename(self.params)
        return os.path.isfile(name)

    def create_pulse(self):
        """
        Creates the time-domain excitation pulse for the incident beam FDTD
        simulation.
        """
        IncidentBeam.logger.info('Creating temporal pulse')
        dt = self.geom_params.timestep
        nt_inc = self.geom_params.size_t_inc
        nt_pulse = self.geom_params.size_t_pulse
        duration = self.inc_beam_params.duration
        freq = self.inc_beam_params.frequency
        self.time_vals = np.linspace(0.0, dt * (nt_inc - 1), num=nt_inc,
                                     endpoint=True)
        # The duration specified in the parameters is the time at which the
        # Gaussian taper is 1/10th of its maximum value.
        sig = duration / freq / (2.0 * np.sqrt(2.0 * np.log(10.0)))  # [s]
        A = 1.0  # Peak amplitude

        # [rad/s] Angular frequency
        om = 2.0 * np.pi * self.inc_beam_params.frequency

        # Temporal pulse - a Gaussian tapered sinusoid
        self.pulse = A * np.exp(-(self.time_vals - (dt * nt_pulse / 2))**2
                                / (sig**2)) * np.exp(1j * om * self.time_vals)
        self.pulse_fft = np.fft.fft(self.pulse)
        self.pulse_freqs = np.fft.fftfreq(self.pulse.size, dt)
        if self.display_params.debug_plots:
            plot_pulse(self)
            plt.pause(0.001)
            plt.show()
        IncidentBeam.logger.info('Done creating temporal pulse')

    def create_inc_excitation(self):
        """
        Creates the Ex and Ey fields which are fed into the incident beam
        FDTD simulation. This function must be called after create_pulse().

        Returns:
            (np.ndarray, np.ndarray): Field values to feed into the incident
                beam simulation. The first item is Ex and the second is Ey.
        """
        IncidentBeam.logger.info('Creating incident excitation')
        # Grab values from the parameters
        dtype_real = self.processing_params.datatype
        theta = self.inc_beam_params.theta_angle
        phi = self.inc_beam_params.phi_angle
        pol = self.inc_beam_params.polarization
        if dtype_real == 'float32':
            dtype_cplx = 'complex64'
        elif dtype_real == 'float64':
            dtype_cplx = 'complex128'
        else:
            raise ValueError('Invalid datatype string')

        # Compute width of the Gaussian weighting
        # The width specified in the parameters is the width at which the
        # Gaussian taper is 1/10th of its maximum value.
        width = self.inc_beam_params.illumination_width * \
            scipy.constants.c / self.inc_beam_params.frequency
        gauss_radius = width / (2.0 * np.sqrt(2.0 * np.log(10.0)))

        # Create spatial frequencies to feed into taper_k
        nt_inc = self.geom_params.size_t_inc
        nx = self.geom_params.size_x
        ny = self.geom_params.size_y
        dx = self.geom_params.cell_size

        kx = 2 * np.pi * np.repeat(np.atleast_2d(np.fft.fftfreq(nx, dx)),
                                   ny, axis=0)
        kx = kx.reshape((ny, nx))
        kx = np.fft.fftshift(kx, axes=1)

        ky = 2 * np.pi * np.repeat(
            np.atleast_2d(np.fft.fftfreq(ny, dx)).transpose(), nx, axis=1)
        ky = ky.reshape((ny, nx))
        ky = np.fft.fftshift(ky, axes=0)

        # These data structures will hold the output of taper_k
        pulse_ex = np.zeros((nt_inc, ny, nx), dtype_cplx)
        pulse_ey = np.zeros((nt_inc, ny, nx), dtype_cplx)

        # Compute wavenumbers of the pulse spectrum
        ang_freqs = 2 * np.pi * self.pulse_freqs
        all_k = ang_freqs / scipy.constants.c  # Dispersion relation

        # Now we have all the wavenumbers in the excitation pulse
        # The pulse_fft weights the contribution of each wavenumber
        scale = 1.0
        for k_idx in trange(nt_inc, desc='Incident excitation'):
            k_val = all_k[k_idx]
            if k_val == 0.0:
                pulse_ex[k_idx, :, :] = np.zeros((ny, nx), dtype_cplx)
                pulse_ey[k_idx, :, :] = np.zeros((ny, nx), dtype_cplx)
            else:
                kspec_ex, kspec_ey = taper_k(k_val, kx, ky, theta, phi,
                                             gauss_radius, pol)
                pulse_ex[k_idx, :, :] = scale * self.pulse_fft[k_idx] \
                    * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kspec_ex)))
                pulse_ey[k_idx, :, :] = scale * self.pulse_fft[k_idx] \
                    * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(kspec_ey)))

        # Do inverse FFT along time axis (axis 0)
        IncidentBeam.logger.info(
            'Performing inverse FFT to get time-domain excitation')
        for y in trange(ny, desc='IFFT'):
            for x in range(nx):
                pulse_ex[:, y, x] = np.fft.ifft(pulse_ex[:, y, x])
                pulse_ey[:, y, x] = np.fft.ifft(pulse_ey[:, y, x])

        # Normalize so that the maximum magnitude is 10.0.
        # This is so that incident beams with the same illumination width
        # will have the same power no matter what nx and ny are.
        # This needs to be done in numba. Using regular numpy tries to allocate
        # some temporary arrays which causes regular computers to run out of
        # memory.
        @numba.njit
        def normalize_excitation(_pulse_ex, _pulse_ey):
            _maxval = 0
            for t in range(nt_inc):
                for y in range(ny):
                    for x in range(nx):
                        if np.abs(_pulse_ex[t, y, x]) > _maxval:
                            _maxval = np.abs(_pulse_ex[t, y, x])
                        if np.abs(_pulse_ey[t, y, x]) > _maxval:
                            _maxval = np.abs(_pulse_ey[t, y, x])
            
            for t in range(nt_inc):
                for y in range(ny):
                    for x in range(nx):
                        _pulse_ex[t, y, x] *= (10.0 / _maxval)
                        _pulse_ey[t, y, x] *= (10.0 / _maxval)

        IncidentBeam.logger.info('Normalizing the time-domain excitation')
        normalize_excitation(pulse_ex, pulse_ey)

        if self.display_params.debug_plots:
            incident_excitation_movie(
                pulse_ex, pulse_ey, self.display_params.display_step)
        IncidentBeam.logger.info('Done creating incident excitation')
        return (pulse_ex, pulse_ey)


@numba.njit
def taper_k(k, k_x, k_y, theta, phi, radius, pol):
    """ Create a tapered beam in the spatial frequency domain.

    This function implements the tapered wave described in the paper,
    "Tapered Wave with a Least Squared Error Property", by Braunisch. It is
    used to create the Ex and Ey field components that are fed into the
    incident beam FDTD simulation. The variable names in this function mimic
    the variable names in the paper.

    Args:
        k (float): The k value of the beam (2pi/lambda).
        k_x (np.ndarray): X dimension spatial frequency values in a
            2D array of shape (size_y, size_x).
        k_y (np.ndarray): Y dimension spatial frequency values in a
            2D array of shape (size_y, size_x).
        theta (float): Incident angle measured from Z axis [deg]
        phi (float): Azimuthal angle measured from X axis [deg]
        radius (float): Radius of the spatial footprint of the Gaussian
            profile [m]
        pol (str): Beam polarization (HH, VV, RHCP, LHCP)

    Returns:
        (np.ndarray, np.ndarray): A 2D array of the tapered beam in the spatial
            frequency domain. First tuple index is the Ex component and the
            second is Ey.
    """
    ny, nx = k_x.shape

    # Compute other components of spatial frequencies
    k_p = np.hypot(k_x, k_y)  # Eqn 4
    k_z = np.conj(np.sqrt((k**2 - k_p**2).astype(np.complex64)))  # Eqn 5
    # Conjugate will give negative imaginary root for k_p > k as desired

    # Compute Cartesian components of the wavenumber
    theta = np.radians(theta)
    phi = np.radians(phi)
    k_ix = (k * np.sin(theta) * np.cos(phi))
    k_iy = (k * np.sin(theta) * np.sin(phi))
    k_ip = np.hypot(k_ix, k_iy)
    k_iz = np.sqrt(k**2 - k_ip**2)

    # This Gaussian weighting produces the beam taper
    # If radius is very large this is equivalent to a plane wave (Eqn 13)
    gauss_weight = ((radius**2) / (4 * np.pi)) \
        * np.exp(-(radius**2) * (((k_x - k_ix)**2) + ((k_y - k_iy)**2)) / 4)

    # Eqn 10
    hor_x = np.zeros(k_x.shape, dtype=np.complex128)
    hor_y = np.zeros(k_x.shape, dtype=np.complex128)

    for j in range(ny):
        for i in range(nx):
            if k_p[j, i] > 0.0:
                hor_x[j, i] = k_y[j, i] / k_p[j, i]
                hor_y[j, i] = -k_x[j, i] / k_p[j, i]
            else:  # k_p[j, i] == 0.0
                hor_x[j, i] = np.sin(phi)
                hor_y[j, i] = -np.cos(phi)

    # Eqn 11
    ver_x = np.zeros(k_x.shape, dtype=np.complex128)
    ver_y = np.zeros(k_x.shape, dtype=np.complex128)
    ver_z = np.zeros(k_x.shape, dtype=np.complex128)

    for j in range(ny):
        for i in range(nx):
            if k_p[j, i] > 0.0:
                ver_x[j, i] = (k_z[j, i] * k_x[j, i]) / (k * k_p[j, i])
                ver_y[j, i] = (k_z[j, i] * k_y[j, i]) / (k * k_p[j, i])
                ver_z[j, i] = k_p[j, i] / k
            else:  # k_p[j, i] == 0.0
                ver_x[j, i] = np.cos(phi)
                ver_y[j, i] = np.sin(phi)
                ver_z[j, i] = 0

    # Compute values of h and v for the central plane wave value k_ip
    if k_ip > 0:
        hor_ix = k_iy / k_ip
        hor_iy = -k_ix / k_ip

        ver_ix = (k_iz * k_ix) / (k * k_ip)
        ver_iy = (k_iz * k_iy) / (k * k_ip)
        ver_iz = k_ip / k
    else:
        hor_ix = np.sin(phi)
        hor_iy = -np.cos(phi)

        ver_ix = np.cos(phi)
        ver_iy = np.sin(phi)
        ver_iz = 0.0

    # From polarization string, create Eh and Ev values (Eqn 16)
    if pol == 'HH':
        Eh = 1
        Ev = 0
    elif pol == 'VV':
        Eh = 0
        Ev = 1
    elif pol == 'RHCP':
        Eh = -1j
        Ev = 1
    elif pol == 'LHCP':
        Eh = 1j
        Ev = 1
    else:
        raise ValueError('Invalid polarization value')

    # Eqn 14-16: Polarization vector
    eh = Eh * ((hor_ix * hor_x) + (hor_iy * hor_y)) \
        + Ev * ((ver_ix * hor_x) + (ver_iy * hor_y))
    ev = Eh * ((hor_ix * ver_x) + (hor_iy * ver_y)) \
        + Ev * ((ver_ix * ver_x) + (ver_iy * ver_y) + (ver_iz * ver_z))

    # Build up the spectral components of the field using Eqn 8 and 1
    x_comp = gauss_weight * ((eh * hor_x) + (ev * ver_x))
    y_comp = gauss_weight * ((eh * hor_y) + (ev * ver_y))
    return (x_comp, y_comp)
