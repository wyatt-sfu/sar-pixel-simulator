import os
from scipy.constants import speed_of_light
import fdtdsarsim as fdtd
import soil_params as soil_params

SEED = 200
num_sims_per_moisture = 256

dirname = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(dirname, "params.json")
params = fdtd.load_from_json(config_file)

# Grab some values from the parameters struct
freq = params.inc_beam_params.frequency
wavelen = speed_of_light / freq
dx = params.geom_params.cell_size
mat_height = params.geom_params.size_z_material * dx

# Mean height of soil (needs to account for surface roughness)
mean_depth = mat_height - (8 * dx)

materials = fdtd.materials.soilmodels.one_layer_gaussian(
    params,
    soil_params.moistures,
    soil_params.sand_frac,
    soil_params.clay_frac,
    soil_params.height_rms,
    soil_params.corr_len,
    mean_depth,
    num_sims_per_moisture,
    SEED,
)

# Comment this line out if you want to just plot the material models
fdtd.run_experiment_mpi(params, materials)

# Materials testing - uncomment this to see the materials
# import matplotlib.pyplot as plt
# for m in materials:
#    fdtd.utils.plotting.materials.plot_materials(m)
#    plt.show()
