"""
This module is used to perform an FDTD simulation of the incident beam.
The beam is recorded and saved to an HDF5 file which can then be fed into the
main simulation.
"""
import concurrent.futures as mt  # mt stands for multithread
import copy
import time

import h5py
import numpy as np
import scipy
from tqdm import trange

import fdtdsarsim.cpu.field_updates as fields
import fdtdsarsim.utils.log as log
from fdtdsarsim.cpu.fdtd_sim import FdtdSimCpu
from fdtdsarsim.parameters import ParameterCollection, inc_filename
from fdtdsarsim.utils.plotting.simulation import SimulationMovie


class IncidentSimCpu(FdtdSimCpu):
    logger = log.getLog(__name__)

    def __init__(self, params: ParameterCollection):
        """ Generate the incident beam data

        This is done by performing an FDTD simulation on the CPU. The Z
        extent is set to a cropped value which is appropriate for the
        simulation.
        """
        params = copy.deepcopy(params)

        # Fix the height to 3 lambda
        lam = scipy.constants.c / params.inc_beam_params.frequency
        params.geom_params.size_z = int(3.0 * lam
                                        / params.geom_params.cell_size)

        super().__init__(params)

    def run(self, pulse_ex: np.ndarray, pulse_ey: np.ndarray):
        """ Run the FDTD simulation and save the incident beam data.

        Args:
            pulse_ex (np.ndarray): Ex component of the excitation to the
                incident beam simulation (nt, ny, nx).
            pulse_ey (np.ndarray): Ey component of the excitation to the
                incident beam simulation (nt, ny, nx).
        """
        IncidentSimCpu.logger.info('Starting the incident beam simulation')
        t0 = time.monotonic()
        filename = inc_filename(self.params)

        # Create numpy arrays to hold field components and CPML data
        self._allocate_arrays()
        self._allocate_phasors()

        with mt.ThreadPoolExecutor(max_workers=self.nthreads) as executor, \
                h5py.File(filename, 'w') as inc_file:
            self._sim_loop(pulse_ex, pulse_ey, executor, inc_file)
            self._write_metadata(inc_file)

        t1 = time.monotonic()
        IncidentSimCpu.logger.info(
            f'Incident beam simulation complete ({t1 - t0:.3f}s)')

    def _sim_loop(self, pulse_ex: np.ndarray, pulse_ey: np.ndarray,
                  executor: mt.ThreadPoolExecutor, inc_file: h5py.File):
        """ Run the incident simulation loop

        Args:
            pulse_ex (np.ndarray): Ex component of the excitation to the
                incident beam simulation (nt, ny, nx).
            pulse_ey (np.ndarray): Ey component of the excitation to the
                incident beam simulation (nt, ny, nx).
            executor (ThreadPoolExecutor): An open and ready to use thread
                pool executor.
            inc_file (h5py.File): An open and writable h5py file object
        """
        # Parameters
        ntp = self.params.geom_params.size_t_inc
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        npml = self.params.geom_params.size_pml
        dx = self.params.geom_params.cell_size
        dt = self.params.geom_params.timestep
        dtype = self.params.processing_params.datatype
        omega = 2.0 * np.pi * self.params.inc_beam_params.frequency
        nt_pulse = self.params.geom_params.size_t_pulse
        disp_step = self.params.display_params.display_step
        disp_movie = self.params.display_params.inc_sim_movie

        # Create datasets in hdf5 file
        ex_dset = inc_file.create_dataset('ex', (ntp, ny, nx), dtype=dtype)
        ey_dset = inc_file.create_dataset('ey', (ntp, ny, nx), dtype=dtype)
        hx_dset = inc_file.create_dataset('hx', (ntp, ny, nx), dtype=dtype)
        hy_dset = inc_file.create_dataset('hy', (ntp, ny, nx), dtype=dtype)

        # Feed-in and read-out planes
        offset = 8
        feed_in = (nz // 2) + (offset // 2)
        read_out = feed_in - (offset // 2)

        # Constants
        e0 = scipy.constants.epsilon_0
        mu0 = scipy.constants.mu_0

        # Compute imaginary component of the input excitation
        pulse_ex_imag = np.imag(pulse_ex)
        pulse_ey_imag = np.imag(pulse_ey)

        # Build up material constants
        # Simulation space is entirely vacuum
        self.Ca = np.ones((nz, ny, nx), dtype)
        self.Cb = dt / (e0 * dx) * np.ones((nz, ny, nx), dtype)
        if dtype == 'float32':
            np_type = np.float32
        elif dtype == 'float64':
            np_type = np.float64
        else:
            raise ValueError('Invalid dtype specifier')
        self.Db = np_type(dt / (mu0 * dx))
        self.dtm0 = np_type(dt / mu0)
        self.dx = np_type(dx)

        # Debug plotting
        if disp_movie:
            if self.params.inc_beam_params.polarization.lower() == 'VV':
                sim_movie = SimulationMovie(self.params, 'H')
            else:
                sim_movie = SimulationMovie(self.params, 'E')

        for t in trange(ntp, desc='Incident time steps'):
            # E field updates
            self.e_field_update(executor)

            # E field CPML updates
            self.e_field_cpml(executor)

            # Inject pulse into the incident field
            if t < nt_pulse:
                self.ex[feed_in, 1:-1, 0:-1] = pulse_ex_imag[t, 1:-1, 0:-1]
                self.ey[feed_in, 1:-1, 0:-1] = pulse_ey_imag[t, 1:-1, 0:-1]

            # H field updates
            self.h_field_update(executor)

            # H field CPML updates
            self.h_field_cpml(executor)

            # Save incident fields to file
            ex_dset[t, :, :] = self.ex[read_out, :, :]
            ey_dset[t, :, :] = self.ey[read_out, :, :]
            hx_dset[t, :, :] = self.hx[read_out, :, :]
            hy_dset[t, :, :] = self.hy[read_out, :, :]

            # Compute incident beam phasors
            fields.phasors_incident(self.ex, self.ey, self.hx, self.hy,
                                    self.ex_ph, self.ey_ph, self.hx_ph,
                                    self.hy_ph, omega, read_out, t, dt, npml)

            if disp_movie and t % disp_step == 0:
                sim_movie.add_frame(self.ex, self.ey, self.hx, self.hy)

        # Save the incident beam phasors to the incident beam HDF5 file
        phasor_grp = inc_file.create_group('phasors')
        phasor_grp.create_dataset('ex_inc_phasor', data=self.ex_ph)
        phasor_grp.create_dataset('ey_inc_phasor', data=self.ey_ph)
        phasor_grp.create_dataset('hx_inc_phasor', data=self.hx_ph)
        phasor_grp.create_dataset('hy_inc_phasor', data=self.hy_ph)

    def _allocate_phasors(self):
        """ Allocate arrays to store the phasor values.
        """
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        npml = self.params.geom_params.size_pml

        nx_ph = nx - 2 * npml
        ny_ph = ny - 2 * npml

        if self.params.processing_params.datatype == 'float32':
            cplx_type = np.complex64
        elif self.params.processing_params.datatype == 'float64':
            cplx_type = np.complex128
        else:
            raise ValueError('Invalid datatype specified in parameters')

        self.ex_ph = np.zeros((ny_ph, nx_ph), cplx_type)
        self.ey_ph = np.zeros((ny_ph, nx_ph), cplx_type)
        self.hx_ph = np.zeros((ny_ph, nx_ph), cplx_type)
        self.hy_ph = np.zeros((ny_ph, nx_ph), cplx_type)

    def _write_metadata(self, inc_file: h5py.File):
        """ Write incident beam metadata to the incident beam file

        Args:
            inc_file (h5py.File): An open and writable h5py file object
        """
        attrs = inc_file.attrs
        proc_params = self.params.processing_params
        inc_params = self.params.inc_beam_params
        geom_params = self.params.geom_params

        attrs['datatype'] = proc_params.datatype

        attrs['frequency'] = inc_params.frequency
        attrs['polarization'] = inc_params.polarization
        attrs['theta_angle'] = inc_params.theta_angle
        attrs['phi_angle'] = inc_params.phi_angle
        attrs['duration'] = inc_params.duration
        attrs['illumination_width'] = inc_params.illumination_width

        attrs['cell_size'] = geom_params.cell_size
        attrs['timestep'] = geom_params.timestep
        attrs['size_t_inc'] = geom_params.size_t_inc
        attrs['size_t_pulse'] = geom_params.size_t_pulse
        attrs['size_x'] = geom_params.size_x
        attrs['size_y'] = geom_params.size_y
        attrs['size_pml'] = geom_params.size_pml
        attrs['courant'] = geom_params.courant

        attrs['incident_complete'] = True
