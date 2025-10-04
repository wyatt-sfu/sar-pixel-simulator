"""
This module contains the base class for performing FDTD simulations on a CPU.
"""
import concurrent.futures as mt  # mt stands for multithread

import numpy as np

from fdtdsarsim.parameters import ParameterCollection
from fdtdsarsim.absorbers import Absorbers
import fdtdsarsim.cpu.field_updates as fields


class FdtdSimCpu:
    def __init__(self, params: ParameterCollection):
        """ Base class for all CPU based FDTD simulations.

        Currently there are two simulations:
         1) Incident beam simulations
         2) Main simulations with material models
        """
        self.params = params

        # Setup the absorbers
        self.absorbers = Absorbers(self.params)

        # Create some arrays to help with dividing work amongst the threads
        self._setup_threading()

    def e_field_update(self, executor):
        # Ex update
        for idx in range(self.nthreads):
            self.futures[idx] = executor.submit(
                fields.ex_update, self.ex, self.hy, self.hz, self.Ca, self.Cb,
                self.thread_idxs_1[idx][0], self.thread_idxs_1[idx][1])
        mt.wait(self.futures, timeout=None, return_when=mt.ALL_COMPLETED)

        # Ey update
        for idx in range(self.nthreads):
            self.futures[idx] = executor.submit(
                fields.ey_update, self.ey, self.hx, self.hz, self.Ca, self.Cb,
                self.thread_idxs_1[idx][0], self.thread_idxs_1[idx][1])
        mt.wait(self.futures, timeout=None, return_when=mt.ALL_COMPLETED)

        # Ez update
        for idx in range(self.nthreads):
            self.futures[idx] = executor.submit(
                fields.ez_update, self.ez, self.hx, self.hy, self.Ca, self.Cb,
                self.thread_idxs_0[idx][0], self.thread_idxs_0[idx][1])
        mt.wait(self.futures, timeout=None, return_when=mt.ALL_COMPLETED)

    def e_field_cpml(self, executor):
        self.cpml_futures[0] = executor.submit(
            fields.cpml_e_down, self.ex, self.ey, self.hx, self.hy,
            self.psiExz_d, self.psiEyz_d, self.absorbers.be_d,
            self.absorbers.ce_d, self.Cb, self.dx)
        self.cpml_futures[1] = executor.submit(
            fields.cpml_e_up, self.ex, self.ey, self.hx, self.hy,
            self.psiExz_u, self.psiEyz_u, self.absorbers.be_u,
            self.absorbers.ce_u, self.Cb, self.dx)
        mt.wait(self.cpml_futures, timeout=None, return_when=mt.ALL_COMPLETED)

        self.cpml_futures[0] = executor.submit(
            fields.cpml_e_left, self.ey, self.ez, self.hy, self.hz,
            self.psiEyx_l, self.psiEzx_l, self.absorbers.be_l,
            self.absorbers.ce_l, self.Cb, self.dx)
        self.cpml_futures[1] = executor.submit(
            fields.cpml_e_right, self.ey, self.ez, self.hy, self.hz,
            self.psiEyx_r, self.psiEzx_r, self.absorbers.be_r,
            self.absorbers.ce_r, self.Cb, self.dx)
        mt.wait(self.cpml_futures, timeout=None, return_when=mt.ALL_COMPLETED)

        self.cpml_futures[0] = executor.submit(
            fields.cpml_e_front, self.ex, self.ez, self.hx, self.hz,
            self.psiExy_f, self.psiEzy_f, self.absorbers.be_f,
            self.absorbers.ce_f, self.Cb, self.dx)
        self.cpml_futures[1] = executor.submit(
            fields.cpml_e_back, self.ex, self.ez, self.hx, self.hz,
            self.psiExy_b, self.psiEzy_b, self.absorbers.be_b,
            self.absorbers.ce_b, self.Cb, self.dx)
        mt.wait(self.cpml_futures, timeout=None, return_when=mt.ALL_COMPLETED)

    def h_field_update(self, executor):
        # Hx update
        for idx in range(self.nthreads):
            self.futures[idx] = executor.submit(
                fields.hx_update, self.hx, self.ey, self.ez, self.Db,
                self.thread_idxs_0[idx][0], self.thread_idxs_0[idx][1])
        mt.wait(self.futures, timeout=None, return_when=mt.ALL_COMPLETED)

        # Hy update
        for idx in range(self.nthreads):
            self.futures[idx] = executor.submit(
                fields.hy_update, self.hy, self.ex, self.ez, self.Db,
                self.thread_idxs_0[idx][0], self.thread_idxs_0[idx][1])
        mt.wait(self.futures, timeout=None, return_when=mt.ALL_COMPLETED)

        # Hz update
        for idx in range(self.nthreads):
            self.futures[idx] = executor.submit(
                fields.hz_update, self.hz, self.ex, self.ey, self.Db,
                self.thread_idxs_1[idx][0], self.thread_idxs_1[idx][1])
        mt.wait(self.futures, timeout=None, return_when=mt.ALL_COMPLETED)

    def h_field_cpml(self, executor):
        self.cpml_futures[0] = executor.submit(
            fields.cpml_h_down, self.ex, self.ey, self.hx, self.hy,
            self.psiHxz_d, self.psiHyz_d, self.absorbers.bh_d,
            self.absorbers.ch_d, self.dtm0, self.dx)
        self.cpml_futures[1] = executor.submit(
            fields.cpml_h_up, self.ex, self.ey, self.hx, self.hy,
            self.psiHxz_u, self.psiHyz_u, self.absorbers.bh_u,
            self.absorbers.ch_u, self.dtm0, self.dx)
        mt.wait(self.cpml_futures, timeout=None, return_when=mt.ALL_COMPLETED)

        self.cpml_futures[0] = executor.submit(
            fields.cpml_h_left, self.ey, self.ez, self.hy, self.hz,
            self.psiHyx_l, self.psiHzx_l, self.absorbers.bh_l,
            self.absorbers.ch_l, self.dtm0, self.dx)
        self.cpml_futures[1] = executor.submit(
            fields.cpml_h_right, self.ey, self.ez, self.hy, self.hz,
            self.psiHyx_r, self.psiHzx_r, self.absorbers.bh_r,
            self.absorbers.ch_r, self.dtm0, self.dx)
        mt.wait(self.cpml_futures, timeout=None, return_when=mt.ALL_COMPLETED)

        self.cpml_futures[0] = executor.submit(
            fields.cpml_h_front, self.ex, self.ez, self.hx, self.hz,
            self.psiHxy_f, self.psiHzy_f, self.absorbers.bh_f,
            self.absorbers.ch_f, self.dtm0, self.dx)
        self.cpml_futures[1] = executor.submit(
            fields.cpml_h_back, self.ex, self.ez, self.hx, self.hz,
            self.psiHxy_b, self.psiHzy_b, self.absorbers.bh_b,
            self.absorbers.ch_b, self.dtm0, self.dx)
        mt.wait(self.cpml_futures, timeout=None, return_when=mt.ALL_COMPLETED)

    def _setup_threading(self):
        nz = self.params.geom_params.size_z
        self.nthreads = self.params.processing_params.threads_per_proc
        self.thread_idxs_0 = divide_work(0, nz - 1, self.nthreads)
        self.thread_idxs_1 = divide_work(1, nz - 1, self.nthreads)
        self.futures = [None] * self.nthreads
        self.cpml_futures = [None] * 2

    def _allocate_arrays(self):
        nx = self.params.geom_params.size_x
        ny = self.params.geom_params.size_y
        nz = self.params.geom_params.size_z
        npml = self.params.geom_params.size_pml
        dtype = self.params.processing_params.datatype

        # Field components
        self.ex = np.zeros((nz, ny, nx), dtype)
        self.ey = np.zeros((nz, ny, nx), dtype)
        self.ez = np.zeros((nz, ny, nx), dtype)
        self.hx = np.zeros((nz, ny, nx), dtype)
        self.hy = np.zeros((nz, ny, nx), dtype)
        self.hz = np.zeros((nz, ny, nx), dtype)

        # Absorbers
        # Up/down
        self.psiExz_d = np.zeros((npml, ny, nx), dtype)
        self.psiEyz_d = np.zeros((npml, ny, nx), dtype)
        self.psiExz_u = np.zeros((npml, ny, nx), dtype)
        self.psiEyz_u = np.zeros((npml, ny, nx), dtype)
        self.psiHxz_d = np.zeros((npml, ny, nx), dtype)
        self.psiHyz_d = np.zeros((npml, ny, nx), dtype)
        self.psiHxz_u = np.zeros((npml, ny, nx), dtype)
        self.psiHyz_u = np.zeros((npml, ny, nx), dtype)

        # Left/right
        self.psiEyx_l = np.zeros((nz, ny, npml), dtype)
        self.psiEzx_l = np.zeros((nz, ny, npml), dtype)
        self.psiEyx_r = np.zeros((nz, ny, npml), dtype)
        self.psiEzx_r = np.zeros((nz, ny, npml), dtype)
        self.psiHyx_l = np.zeros((nz, ny, npml), dtype)
        self.psiHzx_l = np.zeros((nz, ny, npml), dtype)
        self.psiHyx_r = np.zeros((nz, ny, npml), dtype)
        self.psiHzx_r = np.zeros((nz, ny, npml), dtype)

        # Front/back
        self.psiExy_f = np.zeros((nz, npml, nx), dtype)
        self.psiEzy_f = np.zeros((nz, npml, nx), dtype)
        self.psiExy_b = np.zeros((nz, npml, nx), dtype)
        self.psiEzy_b = np.zeros((nz, npml, nx), dtype)
        self.psiHxy_f = np.zeros((nz, npml, nx), dtype)
        self.psiHzy_f = np.zeros((nz, npml, nx), dtype)
        self.psiHxy_b = np.zeros((nz, npml, nx), dtype)
        self.psiHzy_b = np.zeros((nz, npml, nx), dtype)


def divide_work(global_start: int, global_end: int, nthreads: int):
    """ Returns a List[Tuple[start_idx, end_idx]]

    Args:
        global_start: First index of the array to be split up (inclusive)
        global_end: Last index of the array to split up (exclusive)
        nthreads: Total number of threads
    """
    if global_end <= global_start:
        raise ValueError('Start index must be less than end index')
    rough_load = (global_end - global_start) // nthreads
    work_split = []
    start = global_start
    for _ in range(nthreads - 1):
        work_split.append((start, start + rough_load))
        start += rough_load

    work_split.append((start, global_end))
    return work_split
