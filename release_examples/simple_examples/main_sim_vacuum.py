import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

import fdtdsarsim as fdtd

# Can turn on various display options in the json file

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(dirname, 'test_params_hh.json')
    params = fdtd.load_from_json(config_file)
    materials = fdtd.materials.all_vacuum(params)
    fdtd.run_experiment(params, materials)

    # Results filename:
    results_fname = os.path.join(params.processing_params.project_dir,
                                 'results', 'vacuum.hdf5')

