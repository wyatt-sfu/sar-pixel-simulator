import os
import glob

import h5py
import numpy as np

import fdtdsarsim as fdtd
import soil_params as soil_params

dirname = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(dirname, "params.json")
params = fdtd.load_from_json(config_file)
num_sims = 256

# Moistures
moistures = soil_params.moistures
num_moistures = len(soil_params.moistures)

# Get filenames
proj_dir = params.processing_params.project_dir
data_dir = os.path.join(proj_dir, "results")
experiment_fnames = glob.glob(os.path.join(data_dir, "*.hdf5"))

# Check that each experiment has the expected number of files
assert len(experiment_fnames) // num_moistures == num_sims

print(f"Number of moisture values: {num_moistures}")
print(f"Moisture values: {moistures}")

# Grab common metadata from the first hdf5 file
with h5py.File(experiment_fnames[0], "r") as data:
    attrs = data["parameters"].attrs
    inc_ang = attrs["theta_angle"]
    ff_grp = data["farfield"]
    ff_attrs = ff_grp.attrs
    ff_angles = ff_attrs["theta_angles"]
    print(f"Incident angle: {inc_ang}")

# Backscatter angle is negative of the incident angle
back_ang = -inc_ang

# Figure out what index we need to use to grab the right far field data
back_idx = np.squeeze(np.where(ff_angles == back_ang))
print(f"Index of backscatter: {back_idx}")

results = {}
_phasor = np.zeros((num_moistures, num_sims), dtype="complex128")
for mv_idx in range(len(moistures)):
    mv = moistures[mv_idx]
    for file_idx in range(num_sims):
        fname = os.path.join(proj_dir, "results", f"mv_{mv:.2f}_num_{file_idx}.hdf5")
        with h5py.File(fname, "r") as data:
            _phasor[mv_idx, file_idx] = data["farfield"]["e_phi"][back_idx]  # Use e_phi for HH pol, e_theta for VV pol

results["backscatter_phasor"] = _phasor

# Save this data to a npz file for plotting/further processing
outname = os.path.join(proj_dir, "example.npz")
np.savez(outname, **results)
