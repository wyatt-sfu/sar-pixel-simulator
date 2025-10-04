import os
import matplotlib.pyplot as plt
import numpy as np
import h5py

import fdtdsarsim as fdtd

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(dirname, 'test_params_hh.json')
    params = fdtd.load_from_json(config_file)
    materials = fdtd.materials.flat_surface(params, 1.0, 1.0e10)
    fdtd.run_experiment(params, materials)

    # Results filename:
    results_fname = os.path.join(params.processing_params.project_dir,
                                 'results', 'planar_surface.hdf5')

    # Some plotting and post processing
    with h5py.File(results_fname, 'r') as results:
        ff_grp = results['farfield']
        ff_attrs = ff_grp.attrs
        thetas = ff_attrs['theta_angles']
        out_pwr_h = np.squeeze(ff_grp['out_pwr_far_h'][()])
        out_pwr_v = np.squeeze(ff_grp['out_pwr_far_v'][()])

        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True,
                                       figsize=(8, 6))
        ax1.plot(thetas, 10 * np.log10(out_pwr_h))
        ax1.set_title('Far field power (H)')
        ax1.set_xlabel('Theta angle')
        ax1.grid(True)
        ax2.plot(thetas, 10 * np.log10(out_pwr_v))
        ax2.set_title('Far field power (V)')
        ax2.set_xlabel('Theta angle')
        ax2.grid(True)

    plt.show()
