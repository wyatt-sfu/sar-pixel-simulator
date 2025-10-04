"""
This module is used to perform an FDTD simulation on the CPU.
"""
import time
import concurrent.futures as mt  # mt stands for multithread
import os
import dataclasses

import numpy as np
import scipy
import h5py

from fdtdsarsim.cpu.fdtd_sim import FdtdSimCpu
from fdtdsarsim.parameters import (
    ParameterCollection, inc_filename, results_filename
)
import fdtdsarsim.utils.log as log
from fdtdsarsim.utils.plotting.simulation import SimulationMovie
import fdtdsarsim.cpu.field_updates as fields


class MainSimCpu(FdtdSimCpu):
    logger = log.getLog(__name__)

    def __init__(self, params: ParameterCollection, mpi=False):
        """ Performs FDTD simulations on a CPU.
        """
        if mpi:
            # When using MPI we cannot display fancy progress bars because
            # the printing is messed up.
            self.trange = range
        else:
            # Single threaded, so we can use a fancy progress bar
            from tqdm import trange
            self.trange = trange

        super().__init__(params)

    def run(self, Ca: np.ndarray, Cb: np.ndarray, material_name: str,
            material_meta: dict):
        """
        Run the FDTD simulation with the incident beam and material arrays
        provided.

        Equation numbers are from "Computational Electrodynamics: The Finite-
        Difference Time-Domain Method", Taflove, 3rd Ed.

        Args:
            Ca (np.ndarray): Material values used in E field updates (Eq 3.31a)
            Cb (np.ndarray): Material values used in E field updates (Eq 3.31b)
            material_name (str): Name of the material model which is used to
                create the results filename.
            material_meta (dict): Dictionary of material metadata which is
                saved with the results.
        """
        MainSimCpu.logger.info(f'Starting a CPU FDTD simulation '
                               f'of {material_name}')
        t0 = time.monotonic()
        nthreads = self.params.processing_params.threads_per_proc
        dtype = self.params.processing_params.datatype
        self.Ca = Ca.astype(dtype)
        self.Cb = Cb.astype(dtype)

        # Check if the results file already exists. If it does exist, do not
        # run the simulation again.
        out_name = results_filename(self.params, material_name)
        if not os.path.exists(out_name):
            # Create numpy arrays to hold field components and CPML data
            self._allocate_arrays()
            self._allocate_phasors()

            with mt.ThreadPoolExecutor(max_workers=nthreads) as executor, \
                    h5py.File(inc_filename(self.params), 'r') as inc_file:
                self._sim_loop(executor, inc_file)

            # Simulation is done here. Save the results to file.
            self.save_results(material_name, material_meta)
        else:
            MainSimCpu.logger.info(f'{out_name} already exists, skipping')

        t1 = time.monotonic()
        MainSimCpu.logger.info(f'Simulation of {material_name} is '
                               f'complete ({t1 - t0:.3f}s)')

    def _sim_loop(self, executor: mt.ThreadPoolExecutor, inc_file: h5py.File):
        """ Run the incident simulation loop

        Args:
            executor (ThreadPoolExecutor): An open and ready to use thread
                pool executor.
            inc_file (h5py.File): An h5py file object opened in read mode.
        """
        # Parameters
        ntp = self.params.geom_params.size_t_inc
        nt = self.params.geom_params.size_t
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        npml = self.params.geom_params.size_pml
        dx = self.params.geom_params.cell_size
        dt = self.params.geom_params.timestep
        omega = 2.0 * np.pi * self.params.inc_beam_params.frequency
        dtype = self.params.processing_params.datatype
        tfsf_z = self.params.geom_params.tfsf_z
        disp_step = self.params.display_params.display_step
        disp_movie = self.params.display_params.main_sim_movie

        # Constants
        e0 = scipy.constants.epsilon_0
        mu0 = scipy.constants.mu_0

        # Db is the material values used in H field updates (Eqn 3.32b)
        if dtype == 'float32':
            np_type = np.float32
        elif dtype == 'float64':
            np_type = np.float64
        else:
            raise ValueError('Invalid dtype specifier')
        self.Db = np_type(dt / (mu0 * dx))
        self.dtm0 = np_type(dt / mu0)
        self.dx = np_type(dx)
        self.dte0dx = np_type(dt / (e0 * dx))

        # Incident beam datasets stored in HDF5 files
        ex_inc = inc_file['ex']
        ey_inc = inc_file['ey']
        hx_inc = inc_file['hx']
        hy_inc = inc_file['hy']

        # Multithreading for phasor computation
        self.phasor_futures = [None] * 6

        # Debug plotting
        if disp_movie:
            if self.params.inc_beam_params.polarization.lower() == 'VV':
                sim_movie = SimulationMovie(self.params, 'H')
            else:
                sim_movie = SimulationMovie(self.params, 'E')

        for t in self.trange(nt):
            # H field updates
            self.h_field_update(executor)

            # H field CPML updates
            self.h_field_cpml(executor)

            if t < ntp:
                # Inject incident beam E field
                self.hx[tfsf_z, :, :] += self.Db * ey_inc[t, :, :]
                self.hy[tfsf_z, :, :] -= self.Db * ex_inc[t, :, :]

            # E field updates
            self.e_field_update(executor)

            # E field CPML updates
            self.e_field_cpml(executor)

            if t < ntp:
                # Inject incident beam H field
                self.ey[tfsf_z, :, :] += self.dte0dx * hx_inc[t, :, :]
                self.ex[tfsf_z, :, :] -= self.dte0dx * hy_inc[t, :, :]

            if disp_movie and t % disp_step == 0:
                sim_movie.add_frame(self.ex, self.ey, self.hx, self.hy)

            # Compute phasors over the 6 internal faces of the simulation
            # The phasors are computed inside of the PMLs
            self.phasor_futures[0] = executor.submit(
                fields.phasors_x_plane, self.ey, self.ez, self.hy, self.hz,
                self.ey_ph_l, self.ez_ph_l, self.hy_ph_l, self.hz_ph_l, omega,
                npml + 1, t, dt, npml)
            self.phasor_futures[1] = executor.submit(
                fields.phasors_x_plane, self.ey, self.ez, self.hy, self.hz,
                self.ey_ph_r, self.ez_ph_r, self.hy_ph_r, self.hz_ph_r, omega,
                nx - npml - 1, t, dt, npml)

            self.phasor_futures[2] = executor.submit(
                fields.phasors_y_plane, self.ex, self.ez, self.hx, self.hz,
                self.ex_ph_f, self.ez_ph_f, self.hx_ph_f, self.hz_ph_f, omega,
                npml + 1, t, dt, npml)
            self.phasor_futures[3] = executor.submit(
                fields.phasors_y_plane, self.ex, self.ez, self.hx, self.hz,
                self.ex_ph_b, self.ez_ph_b, self.hx_ph_b, self.hz_ph_b, omega,
                ny - npml - 1, t, dt, npml)

            self.phasor_futures[4] = executor.submit(
                fields.phasors_z_plane, self.ex, self.ey, self.hx, self.hy,
                self.ex_ph_d, self.ey_ph_d, self.hx_ph_d, self.hy_ph_d, omega,
                npml + 1, t, dt, npml)
            self.phasor_futures[5] = executor.submit(
                fields.phasors_z_plane, self.ex, self.ey, self.hx, self.hy,
                self.ex_ph_u, self.ey_ph_u, self.hx_ph_u, self.hy_ph_u, omega,
                nz - npml - 1, t, dt, npml)
            mt.wait(self.phasor_futures, timeout=None,
                    return_when=mt.ALL_COMPLETED)

    def save_results(self, material_name, material_meta):
        """ Save the simulation results to an HDF5 file for analysis.

        The following items need to be saved: simulation parameters, material
        metadata and phasors.
        Args:
            material_name (str): Name of the material model which is used to
                create the results filename.
            material_meta (dict): Dictionary of material metadata which is
                saved with the results.
        """
        out_filename = results_filename(self.params, material_name)
        if os.path.exists(out_filename):
            raise RuntimeError('Overwriting a previous simulation. '
                               'This is probably bad!')

        with h5py.File(out_filename, 'w') as out_file, \
                h5py.File(inc_filename(self.params), 'r') as inc_file:
            phasor_grp = out_file.create_group('phasors')
            param_grp = out_file.create_group('parameters')
            material_grp = out_file.create_group('material_metadata')
            inc_phasor_grp = inc_file['phasors']

            phasor_names = [
                'ex_ph_d', 'ey_ph_d', 'hx_ph_d', 'hy_ph_d',
                'ex_ph_u', 'ey_ph_u', 'hx_ph_u', 'hy_ph_u',

                'ey_ph_l', 'ez_ph_l', 'hy_ph_l', 'hz_ph_l',
                'ey_ph_r', 'ez_ph_r', 'hy_ph_r', 'hz_ph_r',

                'ex_ph_f', 'ez_ph_f', 'hx_ph_f', 'hz_ph_f',
                'ex_ph_b', 'ez_ph_b', 'hx_ph_b', 'hz_ph_b'
            ]

            # Write main simulation phasors to output file
            phasor_grp.attrs['nx_phasor'] = self.nx_ph
            phasor_grp.attrs['ny_phasor'] = self.ny_ph
            phasor_grp.attrs['nz_phasor'] = self.nz_ph
            for phasor in phasor_names:
                phasor_grp.create_dataset(phasor, data=getattr(self, phasor))

            # Write incident phasors to output file
            phasor_grp.create_dataset('ex_inc_phasor',
                                      data=inc_phasor_grp['ex_inc_phasor'])
            phasor_grp.create_dataset('ey_inc_phasor',
                                      data=inc_phasor_grp['ey_inc_phasor'])
            phasor_grp.create_dataset('hx_inc_phasor',
                                      data=inc_phasor_grp['hx_inc_phasor'])
            phasor_grp.create_dataset('hy_inc_phasor',
                                      data=inc_phasor_grp['hy_inc_phasor'])

            # Write out parameters as attributes
            for param_type_field in dataclasses.fields(self.params):
                dataclass_obj = getattr(self.params, param_type_field.name)
                for field in dataclasses.fields(dataclass_obj):
                    field_name = field.name
                    field_val = getattr(dataclass_obj, field_name)
                    param_grp.attrs[field_name] = field_val

            # Write out material metadata
            if material_meta is not None:
                for key, val in material_meta.items():
                    material_grp.attrs[key] = val

    def _allocate_phasors(self):
        """ Allocate arrays to store the phasor values.

        The phasors are computed inside of the PML absorbers.
        """
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        npml = self.params.geom_params.size_pml

        self.nx_ph = nx - 2 * npml
        self.ny_ph = ny - 2 * npml
        self.nz_ph = nz - 2 * npml

        if self.params.processing_params.datatype == 'float32':
            cplx_type = np.complex64
        elif self.params.processing_params.datatype == 'float64':
            cplx_type = np.complex128
        else:
            raise ValueError('Invalid datatype specified in parameters')

        self.ex_ph_d = np.zeros((self.ny_ph, self.nx_ph), cplx_type)
        self.ey_ph_d = np.zeros((self.ny_ph, self.nx_ph), cplx_type)
        self.hx_ph_d = np.zeros((self.ny_ph, self.nx_ph), cplx_type)
        self.hy_ph_d = np.zeros((self.ny_ph, self.nx_ph), cplx_type)
        self.ex_ph_u = np.zeros((self.ny_ph, self.nx_ph), cplx_type)
        self.ey_ph_u = np.zeros((self.ny_ph, self.nx_ph), cplx_type)
        self.hx_ph_u = np.zeros((self.ny_ph, self.nx_ph), cplx_type)
        self.hy_ph_u = np.zeros((self.ny_ph, self.nx_ph), cplx_type)

        self.ey_ph_l = np.zeros((self.nz_ph, self.ny_ph), cplx_type)
        self.ez_ph_l = np.zeros((self.nz_ph, self.ny_ph), cplx_type)
        self.hy_ph_l = np.zeros((self.nz_ph, self.ny_ph), cplx_type)
        self.hz_ph_l = np.zeros((self.nz_ph, self.ny_ph), cplx_type)
        self.ey_ph_r = np.zeros((self.nz_ph, self.ny_ph), cplx_type)
        self.ez_ph_r = np.zeros((self.nz_ph, self.ny_ph), cplx_type)
        self.hy_ph_r = np.zeros((self.nz_ph, self.ny_ph), cplx_type)
        self.hz_ph_r = np.zeros((self.nz_ph, self.ny_ph), cplx_type)

        self.ex_ph_f = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
        self.ez_ph_f = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
        self.hx_ph_f = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
        self.hz_ph_f = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
        self.ex_ph_b = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
        self.ez_ph_b = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
        self.hx_ph_b = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
        self.hz_ph_b = np.zeros((self.nz_ph, self.nx_ph), cplx_type)
