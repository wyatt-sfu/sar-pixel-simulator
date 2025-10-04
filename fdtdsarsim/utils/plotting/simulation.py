import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

from fdtdsarsim.parameters import ParameterCollection
import fdtdsarsim.utils.log as log


class SimulationMovie:
    logger = log.getLog(__name__)

    def __init__(self, params: ParameterCollection, field: str = 'E'):
        """ Create a movie of the simulation as it runs.

        The field component to plot can be specified using the field argument.
        A value of 'E' plots the electric field, and a value of 'H' plots the
        magnetic field.
        """
        SimulationMovie.logger.debug('Initializing FDTD simulation movie')
        self.field = field.upper()
        if self.field != 'E' and self.field != 'H':
            raise ValueError('Invalid field specified for plotting')
        self.params = params

        # Grab important values from parameters
        nx = params.geom_params.size_x
        ny = params.geom_params.size_y
        nz = params.geom_params.size_z
        disp_lim = params.display_params.display_limit

        # Plotting slices through the 3D space
        self.x_slice = nx // 2
        self.y_slice = ny // 2
        self.z_slice = nz // 2

        plt.ion()
        self.fig, self.ax = \
            plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

        first_frame_x = np.zeros((nz, ny))
        first_frame_y = np.zeros((nz, nx))
        first_frame_z = np.zeros((ny, nx))

        # Initialize each of the figures
        self.im = np.ndarray((2, 3), dtype=AxesImage)
        self.ax[0, 0].set_title(f'${self.field}_x$ (yz slice)')
        self.ax[0, 0].set_xlabel('Y')
        self.ax[0, 0].set_ylabel('Z')
        self.im[0, 0] = self.ax[0, 0].imshow(
            first_frame_x, vmin=-disp_lim, vmax=disp_lim, cmap='RdBu',
            origin='lower')
        self.ax[0, 1].set_title(f'${self.field}_x$ (xz slice)')
        self.ax[0, 1].set_xlabel('X')
        self.ax[0, 1].set_ylabel('Z')
        self.im[0, 1] = self.ax[0, 1].imshow(
            first_frame_y, vmin=-disp_lim, vmax=disp_lim, cmap='RdBu',
            origin='lower')
        self.ax[0, 2].set_title(f'${self.field}_x$ (xy slice)')
        self.ax[0, 2].set_xlabel('X')
        self.ax[0, 2].set_ylabel('Y')
        self.im[0, 2] = self.ax[0, 2].imshow(
            first_frame_z, vmin=-disp_lim, vmax=disp_lim, cmap='RdBu',
            origin='lower')

        self.ax[1, 0].set_title(f'${self.field}_y$ (yz slice)')
        self.ax[1, 0].set_xlabel('Y')
        self.ax[1, 0].set_ylabel('Z')
        self.im[1, 0] = self.ax[1, 0].imshow(
            first_frame_x, vmin=-disp_lim, vmax=disp_lim, cmap='RdBu',
            origin='lower')
        self.ax[1, 1].set_title(f'${self.field}_y$ (xz slice)')
        self.ax[1, 1].set_xlabel('X')
        self.ax[1, 1].set_ylabel('Z')
        self.im[1, 1] = self.ax[1, 1].imshow(
            first_frame_y, vmin=-disp_lim, vmax=disp_lim, cmap='RdBu',
            origin='lower')
        self.ax[1, 2].set_title(f'${self.field}_y$ (xy slice)')
        self.ax[1, 2].set_xlabel('X')
        self.ax[1, 2].set_ylabel('Y')
        self.im[1, 2] = self.ax[1, 2].imshow(
            first_frame_z, vmin=-disp_lim, vmax=disp_lim, cmap='RdBu',
            origin='lower')

        plt.show()

    def add_frame(self, ex, ey, hx, hy):
        z0 = 376.730313668  # [Ohms] Impedance of free space
        if self.field == 'E':
            self.im[0, 0].set_data(ex[:, :, self.x_slice])
            self.im[0, 1].set_data(ex[:, self.y_slice, :])
            self.im[0, 2].set_data(ex[self.z_slice, :, :])
            self.im[1, 0].set_data(ey[:, :, self.x_slice])
            self.im[1, 1].set_data(ey[:, self.y_slice, :])
            self.im[1, 2].set_data(ey[self.z_slice, :, :])
        else:
            self.im[0, 0].set_data(hx[:, :, self.x_slice] * z0)
            self.im[0, 1].set_data(hx[:, self.y_slice, :] * z0)
            self.im[0, 2].set_data(hx[self.z_slice, :, :] * z0)
            self.im[1, 0].set_data(hy[:, :, self.x_slice] * z0)
            self.im[1, 1].set_data(hy[:, self.y_slice, :] * z0)
            self.im[1, 2].set_data(hy[self.z_slice, :, :] * z0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
