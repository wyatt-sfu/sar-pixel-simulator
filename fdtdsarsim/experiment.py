"""
An FDTD experiment is a series (or single) FDTD simulation using a common
incident beam, but a collection of different material models.
"""
import os
from typing import Iterable

import matplotlib.pyplot as plt

from .incidentbeam import IncidentBeam
from .materials.materialmodel import MaterialModel
from .materials.materials import compute_material_arrays
from .parameters import ParameterCollection, set_geom_params
from .processresults import process_all_results
from .utils.log import defaultLogSetup
from .utils.plotting.materials import plot_materials


def get_fdtd_simulator(params: ParameterCollection):
    """
    Returns the FDTD simulator class based on the processing type specified in
    the parameters. The purpose of this is to be careful with which submodules
    are imported. Importing the gpu simulator on a machine that does not have
    pycuda will throw an exception.
    """
    proc_type = params.processing_params.processing_type
    proc_type = proc_type.lower()
    if proc_type == 'cpu':
        from fdtdsarsim.cpu.main_sim import MainSimCpu
        fdtd = MainSimCpu
    elif proc_type == 'gpu':
        # Set an environment variable so the right GPU is used
        os.environ['CUDA_DEVICE'] = str(params.processing_params.gpu_num)
        from fdtdsarsim.gpu.main_sim import MainSimGpu
        fdtd = MainSimGpu
    else:
        raise ValueError('Invalid processing type')
    return fdtd


def run_experiment(params: ParameterCollection,
                   materials: Iterable[MaterialModel]):
    """
    Run an FDTD simulation for each material model in the materials iterable.

    Sometimes the geometric sizes of the simulation space are not specified in
    the params object. This is useful when reading material values stored in
    files. In this case, the geometry parameters are updated after loading the
    first material model.
    """
    defaultLogSetup()
    plt.ion()  # Turns on interactive mode so that plots are non-blocking

    if params.geom_params.size_x == 0 and params.geom_params.size_y == 0 \
            and params.geom_params.size_z == 0:
        geom_specified = False
    else:
        geom_specified = True
    first_material = True

    if geom_specified:
        # Get the simulator class here. We don't want to import the GPU
        # simulator module on a machine without pycuda.
        fdtd = get_fdtd_simulator(params)
        sim = fdtd(params)
        inc_beam = IncidentBeam(params)
        inc_beam.setup()

    for material in materials:
        if not geom_specified and first_material:
            # Get the geometry from the material
            nz_mat = material.conductivity.shape[0]
            ny = material.conductivity.shape[1]
            nx = material.conductivity.shape[2]
            dx = material.cell_size
            set_geom_params(params, nx * dx, ny * dx, nz_mat * dx,
                            params.geom_params.simulation_duration, dx, 'metre')
            first_material = False

            # Now do some delayed setup (create simulator object and incident beam)
            fdtd = get_fdtd_simulator(params)
            sim = fdtd(params)
            inc_beam = IncidentBeam(params)
            inc_beam.setup()

        if params.display_params.material_plot:
            plot_materials(material)
            plt.pause(0.01)
            plt.show()

        Ca, Cb = compute_material_arrays(params, material)

        # Run the actual FDTD simulation
        sim.run(Ca, Cb, material.name, material.metadata)

    process_all_results(params.processing_params.project_dir,
                        params.display_params.debug_plots)
