"""
This module is used to perform an FDTD simulation on a GPU.
"""
import time
import os
import dataclasses
import platform
from ctypes import cdll

import numpy as np
import scipy
import h5py
import pycuda.driver as cuda
from pycuda.gpuarray import GPUArray
from pycuda.compiler import SourceModule

from fdtdsarsim.parameters import (
    ParameterCollection, inc_filename, results_filename
)
import fdtdsarsim.utils.log as log
from fdtdsarsim.utils.plotting.simulation import SimulationMovie
from fdtdsarsim.absorbers import Absorbers
from fdtdsarsim.gpu.field_updates import kernel_templates

# Load CUDA shared library
if platform.system() == 'Linux':
    cuda_home = os.environ['CUDA_HOME']
    cudart = cdll.LoadLibrary(os.path.join(cuda_home, 'lib64', 'libcudart.so'))

    # Load Nvidia NPP library
    npp = cdll.LoadLibrary(os.path.join(cuda_home, 'lib64', 'libnpps.so'))
elif platform.system() == 'Windows':
    cuda_path = os.environ['CUDA_PATH']
    cudart = cdll.LoadLibrary(os.path.join(cuda_path, 'bin', 'cudart64_12.dll'))

    # Load Nvidia NPP library
    npp = cdll.LoadLibrary(os.path.join(cuda_path, 'bin', 'npps64_12.dll'))
else:
    raise RuntimeError(f'Unsupported platform: {platform.system()}')


class MainSimGpu:
    logger = log.getLog(__name__)

    def __init__(self, params: ParameterCollection, mpi=False):
        """ Performs FDTD simulations on a GPU.
        """
        if mpi:
            # When using MPI we cannot display fancy progress bars because
            # the printing is messed up.
            self.trange = range
        else:
            # Single threaded, so we can use a fancy progress bar
            from tqdm import trange
            self.trange = trange

        self.params = params
        self.gpu_num = self.params.processing_params.gpu_num

        # Setup the absorbers
        self.absorbers = Absorbers(self.params)

        MainSimGpu.logger.info(
            f'Initializing CUDA context for GPU {self.gpu_num}')
        import pycuda.autoinit  # noqa: F401

        self._init_phasor_sizes()
        self._prepare_cuda_kernels()
        self._allocate_cuda_arrays()
        self._allocate_host_arrays()
        self._init_absorber_arrays()
        self._clear_cuda_arrays()

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
        MainSimGpu.logger.info(f'Starting a GPU FDTD simulation '
                               f'of {material_name} on GPU {self.gpu_num}')
        t0 = time.monotonic()
        dtype = self.params.processing_params.datatype
        self.Ca = Ca.astype(dtype)
        self.Cb = Cb.astype(dtype)

        # Check if the results file already exists. If it does exist, do not
        # run the simulation again.
        out_name = results_filename(self.params, material_name)
        if not os.path.exists(out_name):
            self._clear_cuda_arrays()

            with h5py.File(inc_filename(self.params), 'r') as inc_file:
                self._sim_loop(inc_file)

            # Simulation is done here. Save the results to file.
            self.save_results(material_name, material_meta)
        else:
            MainSimGpu.logger.info(f'{out_name} already exists, skipping')

        t1 = time.monotonic()
        MainSimGpu.logger.info(
            f'Simulation of {material_name} on GPU {self.gpu_num} is complete '
            f'({t1 - t0:.3f}s)')

    def _sim_loop(self, inc_file: h5py.File):
        """ Run the incident simulation loop

        Args:
            executor (ThreadPoolExecutor): An open and ready to use thread
                pool executor.
            inc_file (h5py.File): An h5py file object opened in read mode.
        """
        # Parameters
        freq = self.params.inc_beam_params.frequency
        ntp = self.params.geom_params.size_t_inc
        nt = self.params.geom_params.size_t
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        nx_ph = self.nx_ph
        ny_ph = self.ny_ph
        nz_ph = self.nz_ph
        npml = self.params.geom_params.size_pml
        dx = self.params.geom_params.cell_size
        dt = self.params.geom_params.timestep
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
        omega = np_type(2.0 * np.pi * freq)
        dt_val = np_type(dt)

        # Incident beam datasets stored in HDF5 files
        ex_inc = inc_file['ex']
        ey_inc = inc_file['ey']
        hx_inc = inc_file['hx']
        hy_inc = inc_file['hy']

        # Grids/blocks for the various kernels
        field_block = (256, 1, 1)
        field_grid = ((nx + field_block[0] - 1) // field_block[0],
                      (ny + field_block[1] - 1) // field_block[1],
                      (nz + field_block[2] - 1) // field_block[2])
        inc_block = (256, 1, 1)
        inc_grid = ((nx + inc_block[0] - 1) // inc_block[0],
                    (ny + inc_block[1] - 1) // inc_block[1],
                    1)
        ud_block = (256, 1, 1)
        ud_grid = ((nx + ud_block[0] - 1) // ud_block[0],
                   (ny + ud_block[1] - 1) // ud_block[1],
                   (npml + ud_block[2] - 1) // ud_block[2])
        lr_block = (4, 4, 4)
        lr_grid = ((npml + lr_block[0] - 1) // lr_block[0],
                   (ny + lr_block[1] - 1) // lr_block[1],
                   (nz + lr_block[2] - 1) // lr_block[2])
        fb_block = (4, 4, 4)
        fb_grid = ((nx + fb_block[0] - 1) // fb_block[0],
                   (npml + fb_block[1] - 1) // fb_block[1],
                   (nz + fb_block[2] - 1) // fb_block[2])
        phasor_ud_block = (64, 1, 1)
        phasor_ud_grid = ((nx_ph + phasor_ud_block[0] - 1) // phasor_ud_block[0],
                          (ny_ph + phasor_ud_block[1] - 1) // phasor_ud_block[1],
                          1)
        phasor_lr_block = (1, 64, 1)
        phasor_lr_grid = (1,
                          (ny_ph + phasor_lr_block[1] - 1) // phasor_lr_block[1],
                          (nz_ph + phasor_lr_block[2] - 1) // phasor_lr_block[2])
        phasor_fb_block = (64, 1, 1)
        phasor_fb_grid = ((nx_ph + phasor_fb_block[0] - 1) // phasor_fb_block[0],
                          1,
                          (nz_ph + phasor_fb_block[2] - 1) // phasor_fb_block[2])

        # Transfer material arrays to GPU
        cuda.memcpy_htod(self.Ca_gpu, self.Ca)
        cuda.memcpy_htod(self.Cb_gpu, self.Cb)

        # Debug plotting
        if disp_movie:
            if self.params.inc_beam_params.polarization.lower() == 'VV':
                sim_movie = SimulationMovie(self.params, 'H')
            else:
                sim_movie = SimulationMovie(self.params, 'E')
            ex = np.zeros((nz, ny, nx), dtype=dtype)
            ey = np.zeros((nz, ny, nx), dtype=dtype)
            hx = np.zeros((nz, ny, nx), dtype=dtype)
            hy = np.zeros((nz, ny, nx), dtype=dtype)

        for t in self.trange(nt):
            # H field updates
            self.hx_update.prepared_call(
                field_grid, field_block, self.hx_gpu, self.ey_gpu, self.ez_gpu,
                self.Db)
            self.hy_update.prepared_call(
                field_grid, field_block, self.hy_gpu, self.ex_gpu, self.ez_gpu,
                self.Db)
            self.hz_update.prepared_call(
                field_grid, field_block, self.hz_gpu, self.ex_gpu, self.ey_gpu,
                self.Db)

            # H field CPML updates
            self.cpml_h_down.prepared_call(
                ud_grid, ud_block, self.ex_gpu, self.ey_gpu, self.hx_gpu,
                self.hy_gpu, self.psiHxz_d, self.psiHyz_d, self.bh_d,
                self.ch_d, self.dtm0, self.dx)
            self.cpml_h_up.prepared_call(
                ud_grid, ud_block, self.ex_gpu, self.ey_gpu, self.hx_gpu,
                self.hy_gpu, self.psiHxz_u, self.psiHyz_u, self.bh_u,
                self.ch_u, self.dtm0, self.dx)

            self.cpml_h_left.prepared_call(
                lr_grid, lr_block, self.ey_gpu, self.ez_gpu, self.hy_gpu,
                self.hz_gpu, self.psiHyx_l, self.psiHzx_l, self.bh_l,
                self.ch_l, self.dtm0, self.dx)
            self.cpml_h_right.prepared_call(
                lr_grid, lr_block, self.ey_gpu, self.ez_gpu, self.hy_gpu,
                self.hz_gpu, self.psiHyx_r, self.psiHzx_r, self.bh_r,
                self.ch_r, self.dtm0, self.dx)

            self.cpml_h_front.prepared_call(
                fb_grid, fb_block, self.ex_gpu, self.ez_gpu, self.hx_gpu,
                self.hz_gpu, self.psiHxy_f, self.psiHzy_f, self.bh_f,
                self.ch_f, self.dtm0, self.dx)
            self.cpml_h_back.prepared_call(
                fb_grid, fb_block, self.ex_gpu, self.ez_gpu, self.hx_gpu,
                self.hz_gpu, self.psiHxy_b, self.psiHzy_b, self.bh_b,
                self.ch_b, self.dtm0, self.dx)

            if t < ntp:
                # Inject incident beam E field
                ex_slice = ex_inc[t, :, :].astype(dtype)
                ey_slice = ey_inc[t, :, :].astype(dtype)
                cuda.memcpy_htod(self.ex_inc_gpu, ex_slice)
                cuda.memcpy_htod(self.ey_inc_gpu, ey_slice)
                self.inject_incident_e_field.prepared_call(
                    inc_grid, inc_block, self.hx_gpu, self.hy_gpu,
                    self.ex_inc_gpu, self.ey_inc_gpu, self.Db, tfsf_z)

            # E field updates
            self.ex_update.prepared_call(
                field_grid, field_block, self.ex_gpu, self.hy_gpu, self.hz_gpu,
                self.Ca_gpu, self.Cb_gpu)
            self.ey_update.prepared_call(
                field_grid, field_block, self.ey_gpu, self.hx_gpu, self.hz_gpu,
                self.Ca_gpu, self.Cb_gpu)
            self.ez_update.prepared_call(
                field_grid, field_block, self.ez_gpu, self.hx_gpu, self.hy_gpu,
                self.Ca_gpu, self.Cb_gpu)

            # E field CPML updates
            self.cpml_e_down.prepared_call(
                ud_grid, ud_block, self.ex_gpu, self.ey_gpu, self.hx_gpu,
                self.hy_gpu, self.psiExz_d, self.psiEyz_d, self.be_d,
                self.ce_d, self.Cb_gpu, self.dx)
            self.cpml_e_up.prepared_call(
                ud_grid, ud_block, self.ex_gpu, self.ey_gpu, self.hx_gpu,
                self.hy_gpu, self.psiExz_u, self.psiEyz_u, self.be_u,
                self.ce_u, self.Cb_gpu, self.dx)

            self.cpml_e_left.prepared_call(
                lr_grid, lr_block, self.ey_gpu, self.ez_gpu, self.hy_gpu,
                self.hz_gpu, self.psiEyx_l, self.psiEzx_l, self.be_l,
                self.ce_l, self.Cb_gpu, self.dx)
            self.cpml_e_right.prepared_call(
                lr_grid, lr_block, self.ey_gpu, self.ez_gpu, self.hy_gpu,
                self.hz_gpu, self.psiEyx_r, self.psiEzx_r, self.be_r,
                self.ce_r, self.Cb_gpu, self.dx)

            self.cpml_e_front.prepared_call(
                fb_grid, fb_block, self.ex_gpu, self.ez_gpu, self.hx_gpu,
                self.hz_gpu, self.psiExy_f, self.psiEzy_f, self.be_f,
                self.ce_f, self.Cb_gpu, self.dx)
            self.cpml_e_back.prepared_call(
                fb_grid, fb_block, self.ex_gpu, self.ez_gpu, self.hx_gpu,
                self.hz_gpu, self.psiExy_b, self.psiEzy_b, self.be_b,
                self.ce_b, self.Cb_gpu, self.dx)

            if t < ntp:
                # Inject incident beam H field
                hx_slice = hx_inc[t, :, :].astype(dtype)
                hy_slice = hy_inc[t, :, :].astype(dtype)
                cuda.memcpy_htod(self.hx_inc_gpu, hx_slice)
                cuda.memcpy_htod(self.hy_inc_gpu, hy_slice)
                self.inject_incident_h_field.prepared_call(
                    inc_grid, inc_block, self.ex_gpu, self.ey_gpu,
                    self.hx_inc_gpu, self.hy_inc_gpu, self.dte0dx, tfsf_z)

            if disp_movie and t % disp_step == 0:
                # This greatly slows down the processing as we have to
                # transfer the fields off the gpu
                cuda.memcpy_dtoh(ex, self.ex_gpu)
                cuda.memcpy_dtoh(ey, self.ey_gpu)
                cuda.memcpy_dtoh(hx, self.hx_gpu)
                cuda.memcpy_dtoh(hy, self.hy_gpu)
                sim_movie.add_frame(ex, ey, hx, hy)

            # Compute phasors
            self.phasors_x_plane.prepared_call(
                phasor_lr_grid, phasor_lr_block, self.ey_gpu, self.ez_gpu,
                self.hy_gpu, self.hz_gpu, self.ey_ph_l_gpu, self.ez_ph_l_gpu,
                self.hy_ph_l_gpu, self.hz_ph_l_gpu, omega, npml + 1, t, dt_val)
            self.phasors_x_plane.prepared_call(
                phasor_lr_grid, phasor_lr_block, self.ey_gpu, self.ez_gpu,
                self.hy_gpu, self.hz_gpu, self.ey_ph_r_gpu, self.ez_ph_r_gpu,
                self.hy_ph_r_gpu, self.hz_ph_r_gpu, omega, nx - npml - 1, t,
                dt_val)

            self.phasors_y_plane.prepared_call(
                phasor_fb_grid, phasor_fb_block, self.ex_gpu, self.ez_gpu,
                self.hx_gpu, self.hz_gpu, self.ex_ph_f_gpu, self.ez_ph_f_gpu,
                self.hx_ph_f_gpu, self.hz_ph_f_gpu, omega, npml + 1, t, dt_val)
            self.phasors_y_plane.prepared_call(
                phasor_fb_grid, phasor_fb_block, self.ex_gpu, self.ez_gpu,
                self.hx_gpu, self.hz_gpu, self.ex_ph_b_gpu, self.ez_ph_b_gpu,
                self.hx_ph_b_gpu, self.hz_ph_b_gpu, omega, ny - npml - 1, t,
                dt_val)

            self.phasors_z_plane.prepared_call(
                phasor_ud_grid, phasor_ud_block, self.ex_gpu, self.ey_gpu,
                self.hx_gpu, self.hy_gpu, self.ex_ph_d_gpu, self.ey_ph_d_gpu,
                self.hx_ph_d_gpu, self.hy_ph_d_gpu, omega, npml + 1, t, dt_val)
            self.phasors_z_plane.prepared_call(
                phasor_ud_grid, phasor_ud_block, self.ex_gpu, self.ey_gpu,
                self.hx_gpu, self.hy_gpu, self.ex_ph_u_gpu, self.ey_ph_u_gpu,
                self.hx_ph_u_gpu, self.hy_ph_u_gpu, omega, nz - npml - 1, t,
                dt_val)

            cudart.cudaDeviceSynchronize()

        # Simulation is done, transfer phasor values off of the GPU
        MainSimGpu.logger.info(
            f'Transferring phasor arrays off of GPU {self.gpu_num}')
        cuda.memcpy_dtoh(self.ex_ph_d, self.ex_ph_d_gpu)
        cuda.memcpy_dtoh(self.ey_ph_d, self.ey_ph_d_gpu)
        cuda.memcpy_dtoh(self.hx_ph_d, self.hx_ph_d_gpu)
        cuda.memcpy_dtoh(self.hy_ph_d, self.hy_ph_d_gpu)
        cuda.memcpy_dtoh(self.ex_ph_u, self.ex_ph_u_gpu)
        cuda.memcpy_dtoh(self.ey_ph_u, self.ey_ph_u_gpu)
        cuda.memcpy_dtoh(self.hx_ph_u, self.hx_ph_u_gpu)
        cuda.memcpy_dtoh(self.hy_ph_u, self.hy_ph_u_gpu)

        cuda.memcpy_dtoh(self.ey_ph_l, self.ey_ph_l_gpu)
        cuda.memcpy_dtoh(self.ez_ph_l, self.ez_ph_l_gpu)
        cuda.memcpy_dtoh(self.hy_ph_l, self.hy_ph_l_gpu)
        cuda.memcpy_dtoh(self.hz_ph_l, self.hz_ph_l_gpu)
        cuda.memcpy_dtoh(self.ey_ph_r, self.ey_ph_r_gpu)
        cuda.memcpy_dtoh(self.ez_ph_r, self.ez_ph_r_gpu)
        cuda.memcpy_dtoh(self.hy_ph_r, self.hy_ph_r_gpu)
        cuda.memcpy_dtoh(self.hz_ph_r, self.hz_ph_r_gpu)

        cuda.memcpy_dtoh(self.ex_ph_f, self.ex_ph_f_gpu)
        cuda.memcpy_dtoh(self.ez_ph_f, self.ez_ph_f_gpu)
        cuda.memcpy_dtoh(self.hx_ph_f, self.hx_ph_f_gpu)
        cuda.memcpy_dtoh(self.hz_ph_f, self.hz_ph_f_gpu)
        cuda.memcpy_dtoh(self.ex_ph_b, self.ex_ph_b_gpu)
        cuda.memcpy_dtoh(self.ez_ph_b, self.ez_ph_b_gpu)
        cuda.memcpy_dtoh(self.hx_ph_b, self.hx_ph_b_gpu)
        cuda.memcpy_dtoh(self.hz_ph_b, self.hz_ph_b_gpu)

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

    def _prepare_cuda_kernels(self):
        """ Apply template parameters and compile the cuda kernels.
        """
        MainSimGpu.logger.info('Compiling CUDA kernels')
        if self.params.processing_params.datatype == 'float32':
            c_float_type = 'float'
            py_struct_type = 'f'
            sincos_func = 'sincosf'
        elif self.params.processing_params.datatype == 'float64':
            c_float_type = 'double'
            py_struct_type = 'd'
            sincos_func = 'sincos'
        else:
            raise ValueError('Invalid datatype field in processing parameters')

        kernel_params = {'NZ': self.params.geom_params.size_z,
                         'NY': self.params.geom_params.size_y,
                         'NX': self.params.geom_params.size_x,
                         'NPML': self.params.geom_params.size_pml,
                         'NX_PH': self.nx_ph,
                         'NY_PH': self.ny_ph,
                         'NZ_PH': self.nz_ph,
                         'FPTYPE': c_float_type,
                         'SINCOS': sincos_func}
        kernels = kernel_templates.substitute(kernel_params)
        source_mod = SourceModule(kernels)

        kernel_list = [
            ('ex_update', 'PPPPP'),
            ('ey_update', 'PPPPP'),
            ('ez_update', 'PPPPP'),
            ('cpml_e_down', f'PPPPPPPPP{py_struct_type}'),
            ('cpml_e_up', f'PPPPPPPPP{py_struct_type}'),
            ('cpml_e_left', f'PPPPPPPPP{py_struct_type}'),
            ('cpml_e_right', f'PPPPPPPPP{py_struct_type}'),
            ('cpml_e_front', f'PPPPPPPPP{py_struct_type}'),
            ('cpml_e_back', f'PPPPPPPPP{py_struct_type}'),
            ('hx_update', f'PPP{py_struct_type}'),
            ('hy_update', f'PPP{py_struct_type}'),
            ('hz_update', f'PPP{py_struct_type}'),
            ('cpml_h_down', f'PPPPPPPP{py_struct_type}{py_struct_type}'),
            ('cpml_h_up', f'PPPPPPPP{py_struct_type}{py_struct_type}'),
            ('cpml_h_left', f'PPPPPPPP{py_struct_type}{py_struct_type}'),
            ('cpml_h_right', f'PPPPPPPP{py_struct_type}{py_struct_type}'),
            ('cpml_h_front', f'PPPPPPPP{py_struct_type}{py_struct_type}'),
            ('cpml_h_back', f'PPPPPPPP{py_struct_type}{py_struct_type}'),
            ('inject_incident_e_field', f'PPPP{py_struct_type}i'),
            ('inject_incident_h_field', f'PPPP{py_struct_type}i'),
            ('phasors_x_plane', f'PPPPPPPP{py_struct_type}ii{py_struct_type}'),
            ('phasors_y_plane', f'PPPPPPPP{py_struct_type}ii{py_struct_type}'),
            ('phasors_z_plane', f'PPPPPPPP{py_struct_type}ii{py_struct_type}')
        ]

        MainSimGpu.logger.info(f'Preparing CUDA kernels (GPU {self.gpu_num})')
        for k in kernel_list:
            kernel_name = k[0]
            arg_types = k[1]
            setattr(self, kernel_name, source_mod.get_function(kernel_name))
            kernel = getattr(self, kernel_name)
            kernel.prepare(arg_types)

    def _allocate_cuda_arrays(self):
        """
        Allocate memory on the GPU containing the data structures required to
        run the simulation.
        """
        MainSimGpu.logger.info(f'Allocating memory on GPU {self.gpu_num}')
        dtype = self.params.processing_params.datatype
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        npml = self.params.geom_params.size_pml
        if dtype == 'float32':
            np_type = np.float32
            cplx_type = np.complex64
        elif dtype == 'float64':
            np_type = np.float64
            cplx_type = np.complex128
        else:
            raise ValueError('Invalid dtype specifier')

        # Field components and material specifications
        nbytes_field = np.dtype(np_type).itemsize * nz * ny * nx
        self.ex_gpu = cuda.mem_alloc(nbytes_field)
        self.ey_gpu = cuda.mem_alloc(nbytes_field)
        self.ez_gpu = cuda.mem_alloc(nbytes_field)
        self.hx_gpu = cuda.mem_alloc(nbytes_field)
        self.hy_gpu = cuda.mem_alloc(nbytes_field)
        self.hz_gpu = cuda.mem_alloc(nbytes_field)
        self.Ca_gpu = cuda.mem_alloc(nbytes_field)
        self.Cb_gpu = cuda.mem_alloc(nbytes_field)

        # Incident beam slices
        nbytes_inc = np.dtype(np_type).itemsize * ny * nx
        self.ex_inc_gpu = cuda.mem_alloc(nbytes_inc)
        self.ey_inc_gpu = cuda.mem_alloc(nbytes_inc)
        self.hx_inc_gpu = cuda.mem_alloc(nbytes_inc)
        self.hy_inc_gpu = cuda.mem_alloc(nbytes_inc)

        # Absorbers
        # Up/down
        nbytes_updown = np.dtype(np_type).itemsize * npml * ny * nx
        self.psiExz_d = cuda.mem_alloc(nbytes_updown)
        self.psiEyz_d = cuda.mem_alloc(nbytes_updown)
        self.psiExz_u = cuda.mem_alloc(nbytes_updown)
        self.psiEyz_u = cuda.mem_alloc(nbytes_updown)
        self.psiHxz_d = cuda.mem_alloc(nbytes_updown)
        self.psiHyz_d = cuda.mem_alloc(nbytes_updown)
        self.psiHxz_u = cuda.mem_alloc(nbytes_updown)
        self.psiHyz_u = cuda.mem_alloc(nbytes_updown)
        self.be_d = cuda.mem_alloc(nbytes_updown)
        self.bh_d = cuda.mem_alloc(nbytes_updown)
        self.ce_d = cuda.mem_alloc(nbytes_updown)
        self.ch_d = cuda.mem_alloc(nbytes_updown)
        self.be_u = cuda.mem_alloc(nbytes_updown)
        self.bh_u = cuda.mem_alloc(nbytes_updown)
        self.ce_u = cuda.mem_alloc(nbytes_updown)
        self.ch_u = cuda.mem_alloc(nbytes_updown)

        # Left/right
        nbytes_leftright = np.dtype(np_type).itemsize * nz * ny * npml
        self.psiEyx_l = cuda.mem_alloc(nbytes_leftright)
        self.psiEzx_l = cuda.mem_alloc(nbytes_leftright)
        self.psiEyx_r = cuda.mem_alloc(nbytes_leftright)
        self.psiEzx_r = cuda.mem_alloc(nbytes_leftright)
        self.psiHyx_l = cuda.mem_alloc(nbytes_leftright)
        self.psiHzx_l = cuda.mem_alloc(nbytes_leftright)
        self.psiHyx_r = cuda.mem_alloc(nbytes_leftright)
        self.psiHzx_r = cuda.mem_alloc(nbytes_leftright)
        self.be_l = cuda.mem_alloc(nbytes_leftright)
        self.bh_l = cuda.mem_alloc(nbytes_leftright)
        self.ce_l = cuda.mem_alloc(nbytes_leftright)
        self.ch_l = cuda.mem_alloc(nbytes_leftright)
        self.be_r = cuda.mem_alloc(nbytes_leftright)
        self.bh_r = cuda.mem_alloc(nbytes_leftright)
        self.ce_r = cuda.mem_alloc(nbytes_leftright)
        self.ch_r = cuda.mem_alloc(nbytes_leftright)

        # Front/back
        nbytes_frontback = np.dtype(np_type).itemsize * nz * npml * nx
        self.psiExy_f = cuda.mem_alloc(nbytes_frontback)
        self.psiEzy_f = cuda.mem_alloc(nbytes_frontback)
        self.psiExy_b = cuda.mem_alloc(nbytes_frontback)
        self.psiEzy_b = cuda.mem_alloc(nbytes_frontback)
        self.psiHxy_f = cuda.mem_alloc(nbytes_frontback)
        self.psiHzy_f = cuda.mem_alloc(nbytes_frontback)
        self.psiHxy_b = cuda.mem_alloc(nbytes_frontback)
        self.psiHzy_b = cuda.mem_alloc(nbytes_frontback)
        self.be_f = cuda.mem_alloc(nbytes_frontback)
        self.bh_f = cuda.mem_alloc(nbytes_frontback)
        self.ce_f = cuda.mem_alloc(nbytes_frontback)
        self.ch_f = cuda.mem_alloc(nbytes_frontback)
        self.be_b = cuda.mem_alloc(nbytes_frontback)
        self.bh_b = cuda.mem_alloc(nbytes_frontback)
        self.ce_b = cuda.mem_alloc(nbytes_frontback)
        self.ch_b = cuda.mem_alloc(nbytes_frontback)

        # Phasors
        # Up/down
        nbytes_ph_ud = np.dtype(cplx_type).itemsize * ny * nx
        self.ex_ph_d_gpu = cuda.mem_alloc(nbytes_ph_ud)
        self.ey_ph_d_gpu = cuda.mem_alloc(nbytes_ph_ud)
        self.hx_ph_d_gpu = cuda.mem_alloc(nbytes_ph_ud)
        self.hy_ph_d_gpu = cuda.mem_alloc(nbytes_ph_ud)
        self.ex_ph_u_gpu = cuda.mem_alloc(nbytes_ph_ud)
        self.ey_ph_u_gpu = cuda.mem_alloc(nbytes_ph_ud)
        self.hx_ph_u_gpu = cuda.mem_alloc(nbytes_ph_ud)
        self.hy_ph_u_gpu = cuda.mem_alloc(nbytes_ph_ud)

        # Left/right
        nbytes_ph_lr = np.dtype(cplx_type).itemsize * nz * ny
        self.ey_ph_l_gpu = cuda.mem_alloc(nbytes_ph_lr)
        self.ez_ph_l_gpu = cuda.mem_alloc(nbytes_ph_lr)
        self.hy_ph_l_gpu = cuda.mem_alloc(nbytes_ph_lr)
        self.hz_ph_l_gpu = cuda.mem_alloc(nbytes_ph_lr)
        self.ey_ph_r_gpu = cuda.mem_alloc(nbytes_ph_lr)
        self.ez_ph_r_gpu = cuda.mem_alloc(nbytes_ph_lr)
        self.hy_ph_r_gpu = cuda.mem_alloc(nbytes_ph_lr)
        self.hz_ph_r_gpu = cuda.mem_alloc(nbytes_ph_lr)

        # Front/back
        nbytes_ph_fb = np.dtype(cplx_type).itemsize * nz * nx
        self.ex_ph_f_gpu = cuda.mem_alloc(nbytes_ph_fb)
        self.ez_ph_f_gpu = cuda.mem_alloc(nbytes_ph_fb)
        self.hx_ph_f_gpu = cuda.mem_alloc(nbytes_ph_fb)
        self.hz_ph_f_gpu = cuda.mem_alloc(nbytes_ph_fb)
        self.ex_ph_b_gpu = cuda.mem_alloc(nbytes_ph_fb)
        self.ez_ph_b_gpu = cuda.mem_alloc(nbytes_ph_fb)
        self.hx_ph_b_gpu = cuda.mem_alloc(nbytes_ph_fb)
        self.hz_ph_b_gpu = cuda.mem_alloc(nbytes_ph_fb)

    def _init_phasor_sizes(self):
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        npml = self.params.geom_params.size_pml
        self.nx_ph = nx - 2 * npml
        self.ny_ph = ny - 2 * npml
        self.nz_ph = nz - 2 * npml

    def _allocate_host_arrays(self):
        """
        Allocate memory on the host (cpu side).
        """
        MainSimGpu.logger.info('Allocating memory on host')
        dtype = self.params.processing_params.datatype
        if dtype == 'float32':
            cplx_type = np.complex64
        elif dtype == 'float64':
            cplx_type = np.complex128
        else:
            raise ValueError('Invalid dtype specifier')

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

    def _clear_cuda_arrays(self):
        """
        Reset all the data in the fields and absorber arrays to 0.
        """
        MainSimGpu.logger.info(f'Zero initializing field arrays on '
                               f'GPU {self.gpu_num}')

        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        npml = self.params.geom_params.size_pml
        dtype = self.params.processing_params.datatype
        if dtype == 'float32':
            cplx_type = 'complex64'
        elif dtype == 'float64':
            cplx_type = 'complex128'
        else:
            raise ValueError('Invalid dtype specifier')

        # Clear arrays that have the same shape as the fields
        field_shape = (nz, ny, nx)
        fields = [
            self.ex_gpu, self.ey_gpu, self.ez_gpu, self.hx_gpu, self.hy_gpu,
            self.hz_gpu, self.Ca_gpu, self.Cb_gpu
        ]
        for f in fields:
            GPUArray(allocator=None, shape=field_shape, dtype=dtype,
                     gpudata=f).fill(0.0)

        # Clear incident beam arrays
        inc_shape = (1, ny, nx)
        inc_arrays = [
            self.ex_inc_gpu, self.ey_inc_gpu, self.hx_inc_gpu, self.hy_inc_gpu
        ]
        for i in inc_arrays:
            GPUArray(allocator=None, shape=inc_shape, dtype=dtype,
                     gpudata=i).fill(0.0)

        # Clear CPML up/down arrays
        updown_shape = (npml, ny, nx)
        updown_arrays = [
            self.psiExz_d, self.psiEyz_d, self.psiExz_u, self.psiEyz_u,
            self.psiHxz_d, self.psiHyz_d, self.psiHxz_u, self.psiHyz_u
        ]
        for u in updown_arrays:
            GPUArray(allocator=None, shape=updown_shape, dtype=dtype,
                     gpudata=u).fill(0.0)

        # Clear CPML left/right arrays
        leftright_shape = (nz, ny, npml)
        leftright_arrays = [
            self.psiEyx_l, self.psiEzx_l, self.psiEyx_r, self.psiEzx_r,
            self.psiHyx_l, self.psiHzx_l, self.psiHyx_r, self.psiHzx_r
        ]
        for r in leftright_arrays:
            GPUArray(allocator=None, shape=leftright_shape, dtype=dtype,
                     gpudata=r).fill(0.0)

        # Clear CPML front/back arrays
        frontback_shape = (nz, npml, nx)
        frontback_arrays = [
            self.psiExy_f, self.psiEzy_f, self.psiExy_b, self.psiEzy_b,
            self.psiHxy_f, self.psiHzy_f, self.psiHxy_b, self.psiHzy_b
        ]
        for f in frontback_arrays:
            GPUArray(allocator=None, shape=frontback_shape, dtype=dtype,
                     gpudata=f).fill(0.0)

        # Clear phasor up/down arrays
        phasor_shape = (self.ny_ph, self.nx_ph)
        phasor_arrays_ud = [
            self.ex_ph_d_gpu, self.ey_ph_d_gpu, self.hx_ph_d_gpu, self.hy_ph_d_gpu,
            self.ex_ph_u_gpu, self.ey_ph_u_gpu, self.hx_ph_u_gpu, self.hy_ph_u_gpu
        ]
        for p in phasor_arrays_ud:
            GPUArray(allocator=None, shape=phasor_shape, dtype=cplx_type,
                     gpudata=p).fill(0.0 + 1j * 0.0)

        # Clear phasor left/right arrays
        phasor_shape = (self.nz_ph, self.ny_ph)
        phasor_arrays_lr = [
            self.ey_ph_l_gpu, self.ez_ph_l_gpu, self.hy_ph_l_gpu, self.hz_ph_l_gpu,
            self.ey_ph_r_gpu, self.ez_ph_r_gpu, self.hy_ph_r_gpu, self.hz_ph_r_gpu
        ]
        for p in phasor_arrays_lr:
            GPUArray(allocator=None, shape=phasor_shape, dtype=cplx_type,
                     gpudata=p).fill(0.0 + 1j * 0.0)

        # Clear phasor front/back arrays
        phasor_shape = (self.nz_ph, self.nx_ph)
        phasor_arrays_fb = [
            self.ex_ph_f_gpu, self.ez_ph_f_gpu, self.hx_ph_f_gpu, self.hz_ph_f_gpu,
            self.ex_ph_b_gpu, self.ez_ph_b_gpu, self.hx_ph_b_gpu, self.hz_ph_b_gpu
        ]
        for p in phasor_arrays_fb:
            GPUArray(allocator=None, shape=phasor_shape, dtype=cplx_type,
                     gpudata=p).fill(0.0 + 1j * 0.0)

    def _init_absorber_arrays(self):
        """
        Transfer the CPML field constants to the GPU arrays which have already
        been allocated.
        """
        MainSimGpu.logger.info(f'Initializing absorber field constants on '
                               f'GPU {self.gpu_num}')

        # Up/down
        cuda.memcpy_htod(self.be_d, self.absorbers.be_d)
        cuda.memcpy_htod(self.bh_d, self.absorbers.bh_d)
        cuda.memcpy_htod(self.ce_d, self.absorbers.ce_d)
        cuda.memcpy_htod(self.ch_d, self.absorbers.ch_d)
        cuda.memcpy_htod(self.be_u, self.absorbers.be_u)
        cuda.memcpy_htod(self.bh_u, self.absorbers.bh_u)
        cuda.memcpy_htod(self.ce_u, self.absorbers.ce_u)
        cuda.memcpy_htod(self.ch_u, self.absorbers.ch_u)

        # Left/right
        cuda.memcpy_htod(self.be_l, self.absorbers.be_l)
        cuda.memcpy_htod(self.bh_l, self.absorbers.bh_l)
        cuda.memcpy_htod(self.ce_l, self.absorbers.ce_l)
        cuda.memcpy_htod(self.ch_l, self.absorbers.ch_l)
        cuda.memcpy_htod(self.be_r, self.absorbers.be_r)
        cuda.memcpy_htod(self.bh_r, self.absorbers.bh_r)
        cuda.memcpy_htod(self.ce_r, self.absorbers.ce_r)
        cuda.memcpy_htod(self.ch_r, self.absorbers.ch_r)

        # Front/back
        cuda.memcpy_htod(self.be_f, self.absorbers.be_f)
        cuda.memcpy_htod(self.bh_f, self.absorbers.bh_f)
        cuda.memcpy_htod(self.ce_f, self.absorbers.ce_f)
        cuda.memcpy_htod(self.ch_f, self.absorbers.ch_f)
        cuda.memcpy_htod(self.be_b, self.absorbers.be_b)
        cuda.memcpy_htod(self.bh_b, self.absorbers.bh_b)
        cuda.memcpy_htod(self.ce_b, self.absorbers.ce_b)
        cuda.memcpy_htod(self.ch_b, self.absorbers.ch_b)
