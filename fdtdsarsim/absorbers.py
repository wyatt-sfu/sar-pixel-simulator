"""
The absorbers module contains the Absorber class which is used to setup the
arrays required for the CPML absorbers in both the incident and main FDTD
simulations.
"""
import time
import numpy as np

import fdtdsarsim.utils.log as log
from fdtdsarsim.parameters import ParameterCollection


class Absorbers:
    logger = log.getLog(__name__)

    def __init__(self, params: ParameterCollection):
        """ This class holds the arrays required for CPML absorbers.

        The implementation is based off Chapter 7.9 of Taflove, "Computational
        Electrodynamics: The Finite-Difference Time-Domain Method" (3rd Ed).
        Equations referenced in comments refer to this book. These arrays can
        be quickly generated so they are not stored on disk.

        Args:
            geom_params (GeomParams): Geometry paramaters of the FDTD
                simulation space
            run_setup (bool): Run the setup() method in the constructor
        """
        self.geom_params = params.geom_params
        self.display_params = params.display_params
        self.processing_params = params.processing_params
        self.dtype = np.dtype(params.processing_params.datatype)
        self.setup()

    def setup(self):
        """ Creates the absorber arrays.
        """
        Absorbers.logger.info('Starting absorber setup')
        t0 = time.monotonic()

        # Get geometry parameters
        nx = self.geom_params.size_x
        ny = self.geom_params.size_y
        nz = self.geom_params.size_z
        dx = self.geom_params.cell_size
        dt = self.geom_params.timestep
        npml = self.geom_params.size_pml

        # Physical constants (https://physics.nist.gov/cuu/Constants/)
        eta0 = 376.730313668  # Impedance of free space
        eps0 = 8.8541878128E-12  # Vacuum permittivity

        float_eps = np.finfo(self.dtype).eps

        #######################################################################
        # CPML 1D-profile parameters
        kappa_max = 1  # Kappa maximum, do not change from 1

        # alpha_opt = c*eps/w , where w is the size of the scatterer
        # See Berenger, "Numerical reflection from FDTD-PMLs - a comparison of
        # the split PML with the unsplit and CFS PMLs", IEEE Ant.Prop. vol 50,
        # Mar. 2002
        alpha_max = 0.2

        m_kappa = 3  # Exponent for sigma and kappa polynomial scaling
        m_sigma = m_kappa
        m_alpha = 1  # Exponent for alpha polynomial scaling

        sigma_max = 0.8 * (m_sigma + 1) / (eta0 * dx)  # Eqn 7.66

        #######################################################################
        # CPML 1D-profiles (down, left, front)
        # Increasing indices from 0 to npml - 1
        idx1d_inc = np.arange(npml, dtype=self.dtype)
        # Decreasing indices from npml - 1 to 0
        idx1d_dec = npml - 1 - idx1d_inc

        # Alpha profiles. See Eqn 7.79, 7.115c and Eqn 7.119c
        self.alpha_e_dlf = alpha_max \
            * ((idx1d_inc / (npml - 1)) ** m_alpha).astype(self.dtype)
        self.alpha_h_dlf = alpha_max \
            * (((idx1d_inc + 0.5) / (npml - 1)) ** m_alpha).astype(self.dtype)

        # Sigma profiles. See Eqn 7.115a and 7.119a
        self.sigma_e_dlf = sigma_max \
            * ((idx1d_dec / (npml - 1)) ** m_sigma).astype(self.dtype)
        self.sigma_h_dlf = sigma_max \
            * (((idx1d_dec - 0.5) / (npml - 1)) ** m_sigma).astype(self.dtype)

        # Kappa profiles. See Eqn 7.115b and 7.119b
        self.kappa_e_dlf = 1 + (kappa_max - 1) \
            * ((idx1d_dec / (npml - 1)) ** m_kappa).astype(self.dtype)
        self.kappa_h_dlf = 1 + (kappa_max - 1) \
            * (((idx1d_dec + 0.5) / (npml - 1)) ** m_kappa).astype(self.dtype)

        #######################################################################
        # CPML 1D-profiles (up, right, back)

        # Alpha profiles. See Eqn 7.115c and Eqn 7.119c
        self.alpha_e_urb = alpha_max \
            * ((idx1d_dec / (npml - 1)) ** m_alpha).astype(self.dtype)
        self.alpha_h_urb = alpha_max \
            * (((idx1d_dec - 0.5) / (npml - 1)) ** m_alpha).astype(self.dtype)
        self.alpha_h_urb[self.alpha_h_urb < 0.0] = 0.0

        # Sigma profiles. See Eqn 7.115a and 7.119a
        self.sigma_e_urb = sigma_max \
            * ((idx1d_inc / (npml - 1)) ** m_sigma).astype(self.dtype)
        self.sigma_h_urb = sigma_max \
            * (((idx1d_inc + 0.5) / (npml - 1)) ** m_sigma).astype(self.dtype)

        # Kappa profiles. See Eqn 7.115b and 7.119b
        self.kappa_e_urb = 1 + (kappa_max - 1) \
            * ((idx1d_inc / (npml - 1)) ** m_kappa).astype(self.dtype)
        self.kappa_h_urb = 1 + (kappa_max - 1) \
            * (((idx1d_inc + 0.5) / (npml - 1)) ** m_kappa).astype(self.dtype)

        #######################################################################
        # CPML field constants: down
        # Note: for the CPML field contants see equations 7.99 and 7.102
        be_d = np.exp(-(self.sigma_e_dlf / self.kappa_e_dlf +
                      self.alpha_e_dlf) * dt / eps0)
        self.be_d = np.tile(be_d[:, np.newaxis, np.newaxis], [1, ny, nx])
        bh_d = np.exp(-(self.sigma_h_dlf / self.kappa_h_dlf +
                      self.alpha_h_dlf) * dt / eps0)
        self.bh_d = np.tile(bh_d[:, np.newaxis, np.newaxis], [1, ny, nx])

        ce_d = (self.sigma_e_dlf + float_eps) \
            * (np.exp(-(self.sigma_e_dlf / self.kappa_e_dlf + self.alpha_e_dlf) * dt / eps0) - 1) \
            / (self.sigma_e_dlf * self.kappa_e_dlf + (self.kappa_e_dlf**2 * self.alpha_e_dlf) + float_eps)
        self.ce_d = np.tile(ce_d[:, np.newaxis, np.newaxis], [1, ny, nx])
        ch_d = (self.sigma_h_dlf + float_eps) \
            * (np.exp(-(self.sigma_h_dlf / self.kappa_h_dlf + self.alpha_h_dlf) * dt / eps0) - 1) \
            / (self.sigma_h_dlf * self.kappa_h_dlf + (self.kappa_h_dlf**2 * self.alpha_h_dlf) + float_eps)
        self.ch_d = np.tile(ch_d[:, np.newaxis, np.newaxis], [1, ny, nx])

        #######################################################################
        # CPML field constants: up
        be_u = np.exp(-(self.sigma_e_urb / self.kappa_e_urb +
                      self.alpha_e_urb) * dt / eps0)
        self.be_u = np.tile(be_u[:, np.newaxis, np.newaxis], [1, ny, nx])
        bh_u = np.exp(-(self.sigma_h_urb / self.kappa_h_urb +
                      self.alpha_h_urb) * dt / eps0)
        self.bh_u = np.tile(bh_u[:, np.newaxis, np.newaxis], [1, ny, nx])

        ce_u = (self.sigma_e_urb + float_eps) \
            * (np.exp(-(self.sigma_e_urb / self.kappa_e_urb + self.alpha_e_urb) * dt / eps0) - 1) \
            / (self.sigma_e_urb * self.kappa_e_urb + (self.kappa_e_urb**2 * self.alpha_e_urb) + float_eps)
        self.ce_u = np.tile(ce_u[:, np.newaxis, np.newaxis], [1, ny, nx])
        ch_u = (self.sigma_h_urb + float_eps) \
            * (np.exp(-(self.sigma_h_urb / self.kappa_h_urb + self.alpha_h_urb) * dt / eps0) - 1) \
            / (self.sigma_h_urb * self.kappa_h_urb + (self.kappa_h_urb**2 * self.alpha_h_urb) + float_eps)
        self.ch_u = np.tile(ch_u[:, np.newaxis, np.newaxis], [1, ny, nx])

        #######################################################################
        # CPML field constants: left
        self.be_l = np.tile(be_d[np.newaxis, np.newaxis, :], [nz, ny, 1])
        self.bh_l = np.tile(bh_d[np.newaxis, np.newaxis, :], [nz, ny, 1])

        self.ce_l = np.tile(ce_d[np.newaxis, np.newaxis, :], [nz, ny, 1])
        self.ch_l = np.tile(ch_d[np.newaxis, np.newaxis, :], [nz, ny, 1])

        #######################################################################
        # CPML field constants: right
        self.be_r = np.tile(be_u[np.newaxis, np.newaxis, :], [nz, ny, 1])
        self.bh_r = np.tile(bh_u[np.newaxis, np.newaxis, :], [nz, ny, 1])

        self.ce_r = np.tile(ce_u[np.newaxis, np.newaxis, :], [nz, ny, 1])
        self.ch_r = np.tile(ch_u[np.newaxis, np.newaxis, :], [nz, ny, 1])

        #######################################################################
        # CPML field constants: front
        self.be_f = np.tile(be_d[np.newaxis, :, np.newaxis], [nz, 1, nx])
        self.bh_f = np.tile(bh_d[np.newaxis, :, np.newaxis], [nz, 1, nx])

        self.ce_f = np.tile(ce_d[np.newaxis, :, np.newaxis], [nz, 1, nx])
        self.ch_f = np.tile(ch_d[np.newaxis, :, np.newaxis], [nz, 1, nx])

        #######################################################################
        # CPML field constants: back
        self.be_b = np.tile(be_u[np.newaxis, :, np.newaxis], [nz, 1, nx])
        self.bh_b = np.tile(bh_u[np.newaxis, :, np.newaxis], [nz, 1, nx])

        self.ce_b = np.tile(ce_u[np.newaxis, :, np.newaxis], [nz, 1, nx])
        self.ch_b = np.tile(ch_u[np.newaxis, :, np.newaxis], [nz, 1, nx])

        self.check_dtypes()
        t1 = time.monotonic()
        Absorbers.logger.info(f'Absorber setup complete ({t1 - t0:.3f}s)')

    def check_dtypes(self):
        assert self.be_d.dtype == self.dtype
        assert self.bh_d.dtype == self.dtype
        assert self.be_l.dtype == self.dtype
        assert self.bh_l.dtype == self.dtype
        assert self.be_f.dtype == self.dtype
        assert self.bh_f.dtype == self.dtype

        assert self.be_u.dtype == self.dtype
        assert self.bh_u.dtype == self.dtype
        assert self.be_r.dtype == self.dtype
        assert self.bh_r.dtype == self.dtype
        assert self.be_b.dtype == self.dtype
        assert self.bh_b.dtype == self.dtype

        assert self.ce_d.dtype == self.dtype
        assert self.ch_d.dtype == self.dtype
        assert self.ce_l.dtype == self.dtype
        assert self.ch_l.dtype == self.dtype
        assert self.ce_f.dtype == self.dtype
        assert self.ch_f.dtype == self.dtype

        assert self.ce_u.dtype == self.dtype
        assert self.ch_u.dtype == self.dtype
        assert self.ce_r.dtype == self.dtype
        assert self.ch_r.dtype == self.dtype
        assert self.ce_b.dtype == self.dtype
        assert self.ch_b.dtype == self.dtype
