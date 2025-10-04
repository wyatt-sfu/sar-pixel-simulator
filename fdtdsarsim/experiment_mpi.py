"""
An FDTD experiment is a series (or single) FDTD simulation using a common
incident beam, but a collection of different material models.

This module uses MPI to run multiple simulations in parallel.
"""
from typing import Iterable, List
from enum import Enum
import time
import os.path

from mpi4py import MPI
import numpy as np

from .incidentbeam import IncidentBeam
from .materials.materialmodel import MaterialModel
from .materials.materials import compute_material_arrays
from .parameters import ParameterCollection, set_geom_params, results_filename
from .processresults import process_results
from .utils.log import defaultLogSetup
from .experiment import get_fdtd_simulator
import fdtdsarsim.utils.log as log


class WorkerState(Enum):
    IDLE = 0
    BUSY = 1


class WorkerNextAction(Enum):
    CONTINUE = 0
    STOP = 1


class WorkerSimulationStatus(Enum):
    SUCCESS = 0
    FAILURE = 1


def disable_plotting(params: ParameterCollection):
    """ Turns off all the plotting parameters.
    """
    params.display_params.inc_results = False
    params.display_params.inc_sim_movie = False
    params.display_params.debug_plots = False
    params.display_params.main_sim_movie = False
    params.display_params.material_plot = False


def find_idle_worker(worker_state: List[WorkerState]):
    """
    Returns the index of the first idle worker, or None if there are no idle
    workers.
    """
    for idx, state in enumerate(worker_state):
        if state == WorkerState.IDLE:
            return idx

    return None


def check_responses(requests: List[MPI.Request],
                    worker_state: List[WorkerState]):
    """
    Checks to see if any of the responses from worker nodes have arrived.
    These responses indicate that the worker has completed the current
    simulation and can accept a new material model.
    """
    logger = log.getLog(__name__)
    status = MPI.Status()
    for r in requests:
        complete, obj = r.test(status)
        if complete:
            # This response has been received
            proc_idx = status.Get_source()
            worker_idx = proc_idx - 1
            requests.remove(r)

            logger.info(f'Worker index {worker_idx} is now idle')
            if obj == WorkerSimulationStatus.SUCCESS:
                worker_state[worker_idx] = WorkerState.IDLE
            else:
                raise RuntimeError('Worker process failed')


def run_experiment_mpi(params: ParameterCollection,
                       materials: Iterable[MaterialModel]):
    """
    Run an FDTD simulation for each material model in the materials iterable.

    Sometimes the geometric sizes of the simulation space are not specified in
    the params object. This is useful when reading material values stored in
    files. In this case, the geometry parameters are updated after loading the
    first material model.
    """
    defaultLogSetup()
    logger = log.getLog(__name__)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    n_workers = world_size - 1

    # Disable all plotting for MPI processing
    disable_plotting(params)

    if world_size < 2:
        raise RuntimeError('Must have an MPI world size >= 2')

    if rank == 0:
        # Master process. This process does not perform any FDTD simulations.
        # It's role is to distribute work amongst the worker processes.
        logger.info('Running in MPI mode. All plotting is disabled.')

        # If the processing is done on the GPU we need to check that the number
        # of GPUs and the number of workers line up.
        if params.processing_params.processing_type == 'gpu':
            gpu_num = params.processing_params.gpu_num
            if (n_workers != 1 and isinstance(gpu_num, int)) or \
                    len(gpu_num) != n_workers:
                raise RuntimeError('Number of workers must equal the number '
                                   'of GPUs available')

        if params.geom_params.size_x == 0 and params.geom_params.size_y == 0 \
                and params.geom_params.size_z == 0:
            geom_specified = False
        else:
            geom_specified = True
        first_material = True

        if geom_specified:
            inc_beam = IncidentBeam(params)
            inc_beam.setup()

            logger.info('Broadcasting parameter structure to workers')
            params = comm.bcast(params, root=0)

        # Create a list to track the state of each worker process
        worker_status = [WorkerState.IDLE] * n_workers
        pending_requests = []

        for material in materials:
            # If the result file corresponding to this material model already
            # exists than just skip the rest of this.
            if os.path.exists(results_filename(params, material.name)):
                logger.info(f'Skipping {material.name}, simulation was already completed')
                continue

            if not geom_specified and first_material:
                # Get the geometry from the material
                logger.info('Configuring geometry based on first material')
                nz_mat = material.conductivity.shape[0]
                ny = material.conductivity.shape[1]
                nx = material.conductivity.shape[2]
                dx = material.cell_size
                set_geom_params(
                    params,
                    nx * dx,
                    ny * dx,
                    nz_mat * dx,
                    params.geom_params.simulation_duration,
                    dx,
                    "metre",
                )

                logger.info('Broadcasting parameter structure to workers')
                params = comm.bcast(params, root=0)

                first_material = False
                inc_beam = IncidentBeam(params)
                inc_beam.setup()

            Ca, Cb = compute_material_arrays(params, material)

            # Check if any workers are idle
            worker_idx = find_idle_worker(worker_status)
            if worker_idx is None:
                logger.info('All workers are busy, waiting for an idle worker')

            while worker_idx is None:
                time.sleep(1.0)
                check_responses(pending_requests, worker_status)
                worker_idx = find_idle_worker(worker_status)

            logger.info(f'Starting next simulation on worker {worker_idx}')
            # Update the worker status list to indicate this worker is busy
            worker_status[worker_idx] = WorkerState.BUSY

            # Tell the worker to expect another simulation run
            proc_idx = worker_idx + 1
            comm.ssend(WorkerNextAction.CONTINUE, proc_idx)

            # Send the material arrays to the worker
            logger.info(f'Sending material model {material.name} to '
                        f'worker {worker_idx}')
            comm.Ssend(Ca, proc_idx)
            comm.Ssend(Cb, proc_idx)
            comm.ssend(material.name, proc_idx)
            comm.ssend(material.metadata, proc_idx)

            # Run a non-blocking receive to wait for a signal back from
            # the worker that it is finished running
            pending_requests.append(comm.irecv(source=proc_idx))

        # Once all the materials are complete, send a message to the workers
        # telling them to stop once they have completed the current simulation
        logger.info('Sending stop message to all workers')
        for i in range(n_workers):
            proc_idx = i + 1
            comm.ssend(WorkerNextAction.STOP, proc_idx)
    else:
        # Do FDTD on worker processes
        # Get the simulator class here. We don't want to import the GPU
        # simulator module on a machine without pycuda.
        params = comm.bcast(params, root=0)
        worker_idx = rank - 1

        # If the gpu_num field is a list, replace the list with the gpu_num
        # for this specific worker.
        if isinstance(params.processing_params.gpu_num, list):
            params.processing_params.gpu_num = \
                params.processing_params.gpu_num[worker_idx]
        fdtd = get_fdtd_simulator(params)
        sim = fdtd(params, mpi=True)

        # Allocate material arrays for this worker process
        nz = params.geom_params.size_z
        ny = params.geom_params.size_y
        nx = params.geom_params.size_x
        dtype = params.processing_params.datatype
        Ca = np.zeros((nz, ny, nx), dtype=dtype)
        Cb = np.zeros((nz, ny, nx), dtype=dtype)

        while True:
            next_action = comm.recv(source=0)
            if next_action == WorkerNextAction.STOP:
                logger.info(f'Worker {worker_idx} received stop message')
                return
            else:
                comm.Recv(Ca, source=0)
                comm.Recv(Cb, source=0)
                material_name = comm.recv(source=0)
                material_meta = comm.recv(source=0)
                logger.info(f'Worker {worker_idx} received next material '
                            f'model ({material_name})')
                sim.run(Ca, Cb, material_name, material_meta)
                logger.info(f'Worker {worker_idx} processing results for '
                            f'model {material_name}')
                process_results(results_filename(params, material_name))
                logger.info(f'Worker {worker_idx} has completed '
                            f'model {material_name}')
                comm.ssend(WorkerSimulationStatus.SUCCESS, dest=0)

    if rank == 0:
        logger.info('All simulations are complete!')
