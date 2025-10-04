# sar-pixel-simulator
A specialized FDTD implementation used for simulating the radar response from realistic near Earth surface volumes (such as soil, snow, and vegetation). The simulator is described in the following paper,
> W. Gronnemose and B. Rabus, "3-D FDTD Framework for Simulating SAR Imagery of Realistic Near Earth Surface Volumes (Soil, Snow, and Vegetation)," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-11, 2024.

## Installing
The simulator package can be installed using the command `pip install <path-to-sar-pixel-sim>`.

There are two optional install options, `cuda` and `mpi`. For large simulation work, both of these options should be used. 

Install with just cuda: `pip install <path-to-sar-pixel-sim>[cuda]`

Install with just MPI: `pip install <path-to-sar-pixel-sim>[mpi]`

Install with both: `pip install <path-to-sar-pixel-sim>[cuda,mpi]`

*Note that whitespace in the install options is a problem, i.e. `[cuda, mpi]` will not work with pip.*

Using the mpi4py requires an MPI install. See https://pypi.org/project/mpi4py/ for instructions. When running a CUDA based simulation, the terminal you are in must have nvcc and a C++ compiler on the path. On Windows this means running Python inside of a "Developer Powershell" instance.

## FDTD Experiments
In the context of this simulator, an "experiment" is a series of simulation runs in which the simulator parameters remain constant, but the material models under investigation change. There are two files with different functions for running experiments:
- `experiment.py`: Runs a single simulation at once.
- `experiment_mpi.py`: This can run multiple simulations at once using MPI. Useful for running on large servers or clusters.

An experiment needs only two arguments:
1) An iterable of material models. Using this abstraction means the FDTD simulator does not need to know how the material models are generated.
2) A data class containing the simulation parameters (e.g. polarization, frequency, etc).

## Simulator Parameters
The simulator parameters are stored in a JSON file, and all the expected fields are defined in `parameters.py`. An example params.json file is:
```
{
    "processing_type": "gpu",
    "gpu_num": [0, 1, 2, 3, 4, 5, 6, 7],
    "threads_per_proc": 6,
    "datatype": "float32",
    "project_dir": "/shared/users/wgronnem/example_experiment/",

    "frequency_hz": 5.4e9,
    "polarization": "HH",
    "theta_angle_deg": 20.0,
    "phi_angle_deg": 0.0,
    "incident_duration": 10.0,
    "illumination_width": 12.0,
    "generate_new": false,

    "size_x": 16.0,
    "size_y": 16.0,
    "size_z_material": 3.0,
    "simulation_duration": 18.0,
    "cell_size": 0.021,
    "unit": "wavelength",
    
    "material_fraction": 0.7,

    "disp_inc_results": false,
    "disp_inc_sim_movie": false,
    "disp_debug": false,
    "disp_main_sim_movie": false,
    "disp_materials": false
}
```

### Simulation Geometry
The simulation geometry can be specified in either wavelengths of metres.
As well, if the geometry is not known when the parameters are defined, they can be left out of the params.json file. This is useful if you have material models saved to a file.
In this case, the `run_experiment` function will use the geometry of the first material model.

## Material Models
The material model structure is defined in the file `materialmodel.py`. It requires five values:
- relative permittivity of each cell in the discretized simulation space
- conductivity of each cell in the discretized simulation space
- the cell side length
- a unique name
- arbitrary metadata that will be saved in the output file

## Simulation Output File Structure
The output of the simulator is defined by the `project_dir` parameter.
The `project_dir` folder will have two sub-directories created in it:
1. `incident`: Incident beam file is generated here.
2. `results`: An output file for each material model in the experiment is placed in this folder.

## Incident Beams
Generating the incident beams can be time consuming. However, usually you only need one or two different incident beams for many different experiments. It is a good idea to generate the incident beam folder once, and then symlink into future experiments to reduce both the time and the amount of disk space used.

## Running MPI Experiments
The following is an example of running the script `runsim.py` on a machine with 8 GPU's. We need 9 processes: one for the manager and eight for each simulation worker. The command to run this would be:

`mpiexec -n 9 python -m mpi4py runsim.py`

## Suggested Experiment Script Setup
I suggest putting all the scripts to run each experiment in their own folder. Typically when running an experiment on a compute server an experiment folder contains the following five files:
1. `runsim.py`: Setup the material models and call `run_experiment_mpi`.
2. `params.json`: Simulation parameters.
3. `run_sim.sh`: A convenience script containing the mpiexec command.
4. `gather.py`: Once the simulation is complete, we want to gather only the numbers required for post-processing into a single file. We can then download this onto a personal computer for analysis.
5. `analyze.py`: Plots, analysis, etc on a personal computer.

## Simulation Result Files
The simulation results are stored in HDF5 files. The contents of the files can be easily explored using the HDFView software (https://www.hdfgroup.org/download-hdfview/).