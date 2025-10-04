"""
The parameters module contains data classes to store the parameters required to
run a FDTD simulation.
"""
import json
import os.path
import dataclasses
from dataclasses import dataclass
from typing import List, Union
import scipy.constants

import fdtdsarsim.utils.log as log
from .utils.rounding import round_to_odd


@dataclass
class ProcessingParams:
    """Data class containing file locations required to run an experiment."""

    processing_type: str = 'cpu'  #: CPU vs GPU processing
    gpu_num: Union[int, List[int]] = 0  #: Choose which GPU to run on
    threads_per_proc: int = 1  #: Number of CPU threads per process
    datatype: str = 'float32'  #: Data type to use in simulation
    project_dir: str = ''  #: Path to the project file storage

    def __str__(self):
        return _stringify_dataclass_members(self)


@dataclass
class IncBeamParams:
    """Data class containing the parameters which define an incident beam."""

    frequency: float = 0  #: Center frequency of the incident beam [Hz]
    polarization: str = 0  #: Incident beam polarization (HH, VV, RHCP, LHCP)
    theta_angle: float = 0  #: Incident angle measured from Z axis [deg]
    phi_angle: float = 0  #: Azimuthal angle measured from X axis [deg]

    duration: float = 0
    """ Duration of the temporal Gaussian tapered pulse [vac. wavelengths]"""

    illumination_width: float = 0
    """
    Width (diameter) of the spatial footprint of the Gaussian tapered beam.
    This is the width at which the Gaussian function drops to 1/10th of its
    maximum value. [vac. wavelengths]
    """

    generate_new: bool = False
    """ Set to True to always generate a new incident beam
    """

    def __str__(self):
        return _stringify_dataclass_members(self)


@dataclass
class GeomParams:
    """Data class containing the parameters which define the FDTD geometry."""

    cell_size: float = 0  #: Side length of a cubic Yee cell [m]
    simulation_duration: float = 0  # Simulation duration [# of temporal wavelengths]
    timestep: float = 0  #: Duration of discrete simulation timestep [s]
    size_t: int = 0  #: Number of timesteps in the full simulation
    size_t_inc: int = 0  #: Number of timesteps in the incident beam
    size_t_pulse: int = 0  # Number of timesteps in the temporal pulse
    size_x: int = 0  #: Number of cells in X dimension
    size_y: int = 0  #: Number of cells in Y dimension
    size_z: int = 0  #: Number of cells in Z dimension
    size_pml: int = 0  #: Number of cells in the absorbers

    size_z_material: int = 0
    """
    Number of cells in Z dimension which is filled by materials. The incident
    beam is fed in directly above this height.
    """

    tfsf_z: int = 0
    """
    Z height of the total-field/scattered-field boundary [cells]
    """

    courant: float = 0  #: Courant factor [unitless]

    def __str__(self):
        return _stringify_dataclass_members(self)


@dataclass
class MaterialParams:
    """
    Data class containing parameters which can control how material arrays
    are created.
    """

    material_fraction: float = 0.7
    """
    Nominal fraction of simulation space that is filled with the material
    under test.
    """

    def __str__(self):
        return _stringify_dataclass_members(self)


@dataclass
class DisplayParams:
    """
    Data class which customizes which movies or plots will be shown to the user
    during simulations.
    """

    display_limit: float = 0.0  #: Field amplitude saturation limit for movies
    display_step: int = 10  #: Number of timesteps between movie frames
    inc_results: bool = False  #: Display the incident beam simulation results
    inc_sim_movie: bool = False  #: Show movie of the incident beam simulation
    debug_plots: bool = False  #: Display detailed debug plots
    main_sim_movie: bool = False  #: Show/create a movie of the main simulation
    material_plot: bool = False  #: Plot the material model under investigation

    def __str__(self):
        return _stringify_dataclass_members(self)


@dataclass
class ParameterCollection:
    """
    This class holds all the parameters required to run an FDTD simulation.
    """

    processing_params: ProcessingParams = \
        dataclasses.field(default_factory=ProcessingParams)
    inc_beam_params: IncBeamParams = \
        dataclasses.field(default_factory=IncBeamParams)
    geom_params: GeomParams = dataclasses.field(default_factory=GeomParams)
    material_params: MaterialParams = \
        dataclasses.field(default_factory=MaterialParams)
    display_params: DisplayParams = \
        dataclasses.field(default_factory=DisplayParams)

    def __str__(self):
        val = ''
        for par_obj in dataclasses.fields(self):
            val += f'{getattr(self, par_obj.name).__str__()}\n'
        return val


def _stringify_dataclass_members(dataclass_obj):
    """Used for printing out the values in the parameter classes."""
    if not dataclasses.is_dataclass(dataclass_obj):
        raise ValueError('Can only print a dataclass')

    val = f'{type(dataclass_obj).__name__}:\n'
    for field in dataclasses.fields(dataclass_obj):
        val += f'   {field.name}: {getattr(dataclass_obj, field.name)}\n'
    return val


def load_from_json(filename: str):
    """
    Reads user parameters from a JSON file and returns a ParameterCollection
    object. Geometry parameters can be optionally defined in the JSON file. If
    they are not found in the JSON they can be defined later to give the user
    full flexibility in defining materials.

    Returns:
        ParameterCollection: The parameter collection specified in the JSON
    """
    logger = log.getLog(__name__)
    logger.info(f'Loading parameters from the file {filename}')
    pc = ParameterCollection()
    proc = pc.processing_params
    inc = pc.inc_beam_params
    mat = pc.material_params
    disp = pc.display_params

    with open(filename, 'r') as param_file:
        file_contents = param_file.read()

    user_params = json.loads(file_contents)

    # Processing parameters
    proc.processing_type = user_params['processing_type']
    if 'gpu_num' in user_params:
        proc.gpu_num = user_params['gpu_num']
    else:
        proc.gpu_num = 0
    proc.threads_per_proc = user_params['threads_per_proc']
    proc.datatype = user_params['datatype']
    proc.project_dir = user_params['project_dir']

    # Incident beam parameters
    inc.frequency = user_params['frequency_hz']
    inc.polarization = user_params['polarization']
    inc.theta_angle = user_params['theta_angle_deg']
    inc.phi_angle = user_params['phi_angle_deg']
    inc.duration = user_params['incident_duration']
    inc.illumination_width = user_params['illumination_width']
    inc.generate_new = user_params['generate_new']

    # Material parameters
    mat.material_fraction = user_params['material_fraction']

    # Display parameters
    disp.display_limit = 0.5  # TODO: Don't hardcode this
    disp.display_step = 10
    disp.inc_results = user_params['disp_inc_results']
    disp.inc_sim_movie = user_params['disp_inc_sim_movie']
    disp.debug_plots = user_params['disp_debug']
    disp.main_sim_movie = user_params['disp_main_sim_movie']
    disp.material_plot = user_params['disp_materials']

    # Always read this timing parameter from the JSON, even in cases with
    # delayed geometry setup
    pc.geom_params.simulation_duration = user_params["simulation_duration"]

    # Geometry parameters
    if (
        'size_x' in user_params
        and 'size_y' in user_params
        and 'size_z_material' in user_params
        and 'simulation_duration' in user_params
        and 'cell_size' in user_params
        and 'unit' in user_params
    ):
        logger.info('JSON parameter file contains geometry values')
        set_geom_params(
            pc,
            user_params['size_x'],
            user_params['size_y'],
            user_params['size_z_material'],
            user_params['simulation_duration'],
            user_params['cell_size'],
            user_params['unit'],
        )
    else:
        logger.info('JSON parameter file does not contain geometry values')

    return pc


def set_geom_params(
    pc: ParameterCollection,
    size_x: float,
    size_y: float,
    size_z_material: float,
    size_t: float,
    cell_size: float,
    unit: str,
):
    """
    Add the geometry parameters to the ParameterCollection object. This
    function is split from load_from_json for added flexibility around when
    the material values/sizes are defined.

    The value of unit can either be 'metre' or 'wavelength'. This changes how
    the arguments size_x, size_y, size_z and cell_size are interpreted.

    Args:
        pc: Update the geom_params in these parameters
        size_x: Size of the simulation space in x dimension
        size_y: Size of the simulation space in y dimension
        size_z_material: Size of the simulation space filled with materials
        size_t: Number of temporal wavelengths to run the simulation for
        cell_size: Side length of a cubic Yee cell
        unit: Specifies the units of the previous arguments
    """
    logger = log.getLog(__name__)
    logger.info('Updating geometry parameters')
    pc.geom_params.courant = 0.5
    pc.geom_params.size_pml = 10
    lam = scipy.constants.c / pc.inc_beam_params.frequency

    if unit.lower() == 'metre' or unit.lower() == 'meter':
        pc.geom_params.cell_size = cell_size
    elif unit.lower() == 'wavelength':
        pc.geom_params.cell_size = lam * cell_size
    else:
        raise ValueError(f'Unknown value of unit: {unit}')

    pc.geom_params.size_x = round_to_odd(size_x // cell_size)
    pc.geom_params.size_y = round_to_odd(size_y // cell_size)
    pc.geom_params.size_z_material = int(size_z_material // cell_size)

    mat_frac = pc.material_params.material_fraction
    pc.geom_params.size_z = round_to_odd(pc.geom_params.size_z_material / mat_frac)
    pc.geom_params.tfsf_z = pc.geom_params.size_z_material + int(
        0.3 * lam / pc.geom_params.cell_size
    )

    # Now set timing values
    pc.geom_params.timestep = (
        pc.geom_params.cell_size * pc.geom_params.courant / scipy.constants.c
    )
    delta_t = pc.geom_params.timestep
    freq = pc.inc_beam_params.frequency
    inc_duration = pc.inc_beam_params.duration
    pc.geom_params.size_t_pulse = round(
        1.2 * inc_duration * lam / (scipy.constants.c * delta_t)
    )
    pc.geom_params.size_t_inc = round(1.2 * pc.geom_params.size_t_pulse)
    pc.geom_params.size_t = round(size_t / delta_t / freq)
    if pc.geom_params.size_t_inc > pc.geom_params.size_t:
        logger.warn('Reducing the number of timesteps in the incident beam')
        pc.geom_params.size_t_inc = pc.geom_params.size_t


def inc_filename(pc: ParameterCollection):
    """Creates and returns the incident beam filename.

    Returns:
        str: The full incident beam filename including path.
    """
    size_x = pc.geom_params.size_x
    size_y = pc.geom_params.size_y
    size_t_inc = pc.geom_params.size_t_inc
    theta = round(pc.inc_beam_params.theta_angle)
    phi = round(pc.inc_beam_params.phi_angle)
    pol = pc.inc_beam_params.polarization
    fname = (
        f'inc_nx{size_x}_ny{size_y}_nt{size_t_inc}'
        + f'_theta{theta}_phi{phi}_{pol}.hdf5'
    )
    return os.path.join(inc_file_dir(pc.processing_params), fname)


def results_filename(pc: ParameterCollection, material_name: str):
    """Creates and returns the results filename.

    Returns:
        str: The full results filename including path.
    """
    fname = material_name + '.hdf5'
    return os.path.join(results_file_dir(pc.processing_params), fname)


def inc_file_dir(proc_params: ProcessingParams):
    """Returns the name of the directory to store the incident beam files in.

    This directory is created if it does not already exist.
    """
    inc_dir = os.path.join(proc_params.project_dir, 'incident')
    if not os.path.exists(inc_dir):
        os.makedirs(inc_dir, exist_ok=True)
    return inc_dir


def results_file_dir(proc_params: ProcessingParams):
    """Returns the name of the directory to store the results files in.

    This directory is created if it does not already exist.
    """
    results_dir = os.path.join(proc_params.project_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    return results_dir
