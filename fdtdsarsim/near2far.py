from typing import Mapping

from tqdm import trange
import numpy as np
import numpy.typing as npt
import scipy.integrate

import fdtdsarsim.utils.log as log


class FarFieldTransform:
    logger = log.getLog(__name__)

    def __init__(self):
        """
        Compute the near to far field transformation for analysis of scattering
        properties. The center of the simulation space is considered the
        origin (0, 0, 0).

        Equation numbers are referencing Taflove, "Computational
        Electrodynamics: The Finite-Difference Time-Domain Method" (3rd Ed).
        """
        # For testing different integration methods this can easily be changed
        # to use the trapezoid method.
        # self.integrate_func = scipy.integrate.trapezoid
        self.integrate_func = scipy.integrate.simpson

    def __call__(self, phasors: Mapping[str, npt.ArrayLike], nx: int, ny: int,
                 nz: int, dx: float, r: float, thetas: np.ndarray,
                 phis: np.ndarray, freq: float):
        """ Run the near to far field transform.

        Args:
            phasors: Collection of near-field phasor values for each internal
                face of the simulation space.
            nx: Number of phasor samples in X dimension
            ny: Number of phasor samples in Y dimension
            nz: Number of phasor samples in Z dimension
            dx: Discrete spacing between phasors
            r: Radius of the far-field location.
            thetas: Latitude angles [deg]
            phis: Longitude angles [deg]
            freq: Frequency of interest [Hz]

        Returns:
            Mapping[str, np.ndarray]: The far-field phasors at the locations
                (r, theta, phi). Each of the output arrays has the shape
                (len(nphi), len(theta))
        """
        # Check that thetas and phis are 1D arrays
        if not (len(thetas.shape) == 1 and len(phis.shape) == 1):
            raise RuntimeError('Theta and phi arguments must be 1D arrays')

        # Setup some constants/values required for the next steps
        eta0 = 376.730313668  # [Ohms]
        # https://physics.nist.gov/cgi-bin/cuu/Value?z0
        c0 = 299792458  # [m/s] https://physics.nist.gov/cgi-bin/cuu/Value?c
        lam = c0 / freq  # [m] Wavelength
        self.wavenum = 2.0 * np.pi / lam  # [rad/m] Wavenumber
        thetas = np.deg2rad(thetas)  # [rad] Latitude angles
        phis = np.deg2rad(phis)  # [rad] Longitude angles
        ntheta = thetas.shape[0]
        nphi = phis.shape[0]
        self.dx = dx

        # Convert the phasors to 64 bit floats and make them class attributes
        for ph_name, ph_arr in phasors.items():
            setattr(self, ph_name, np.asarray(ph_arr, dtype='complex128'))

        # Compute near-field phasor locations
        self.x_loc = np.arange(-(nx - 1) // 2, ((nx - 1) // 2) + 1) * dx
        self.y_loc = np.arange(-(ny - 1) // 2, ((ny - 1) // 2) + 1) * dx
        self.z_loc = np.arange(-(nz - 1) // 2, ((nz - 1) // 2) + 1) * dx

        # Output arrays
        ff_weight = 1j * self.wavenum * np.exp(-1j * self.wavenum * r) \
            / (4.0 * np.pi * r)
        e_theta = np.zeros((nphi, ntheta), dtype='complex128')
        e_phi = np.zeros((nphi, ntheta), dtype='complex128')
        h_theta = np.zeros((nphi, ntheta), dtype='complex128')
        h_phi = np.zeros((nphi, ntheta), dtype='complex128')

        # Loop through each theta/phi point and perform the transform
        for phi_idx in trange(nphi, desc='Phi', leave=False):
            for theta_idx in trange(ntheta, desc='Theta', leave=False):
                phi = phis[phi_idx]
                theta = thetas[theta_idx]
                self.N_theta = 0.0 + 1j * 0.0
                self.N_phi = 0.0 + 1j * 0.0
                self.L_theta = 0.0 + 1j * 0.0
                self.L_phi = 0.0 + 1j * 0.0

                # Integrate phasors over the top face of the simulation space
                self._integrate_z_plane(-self.hy_ph_u, self.hx_ph_u,
                                        self.ey_ph_u, -self.ex_ph_u,
                                        self.z_loc[-1], theta, phi)

                # Assemble far-field values
                e_theta[phi_idx, theta_idx] = \
                    ff_weight * (-self.L_phi - eta0 * self.N_theta)
                e_phi[phi_idx, theta_idx] = \
                    ff_weight * (self.L_theta - eta0 * self.N_phi)
                h_theta[phi_idx, theta_idx] = \
                    ff_weight * (self.N_phi - self.L_theta / eta0)
                h_phi[phi_idx, theta_idx] = \
                    ff_weight * (-self.N_theta - self.L_phi / eta0)

        farfield = {'E_theta': np.squeeze(e_theta),
                    'E_phi': np.squeeze(e_phi),
                    'H_theta': np.squeeze(h_theta),
                    'H_phi': np.squeeze(h_phi)}
        return farfield

    def _integrate_z_plane(self, Jx: np.ndarray, Jy: np.ndarray,
                           Mx: np.ndarray, My: np.ndarray, z_val: float,
                           theta: float, phi: float):
        """
        Integrate phasor electric current and magnetic current over the
        specified Z plane.

        Args:
            Jx: X component of phasor electric current
            Jy: Y component of phasor electric current
            Mx: X component of phasor magnetic current
            My: Y component of phasor magnetic current
            z_val: Z value of the plane [m]
            theta: Latitude angle [rad]
            phi: Longitude angle [rad]
        """
        xx, yy = np.meshgrid(self.x_loc, self.y_loc)
        exp_term = np.exp(1j * self.wavenum
                          * ((xx * np.sin(theta) * np.cos(phi))
                             + (yy * np.sin(theta) * np.sin(phi))
                             + (z_val * np.cos(theta))))

        # Integrands for equations 8.33a, 8.33b
        Nxy_theta = (Jx * np.cos(theta) * np.cos(phi)
                     + Jy * np.cos(theta) * np.sin(phi)) * exp_term
        Nxy_phi = (-Jx * np.sin(phi) + Jy * np.cos(phi)) * exp_term

        # Integrands for equations 8.34a, 8.34b
        Lxy_theta = (Mx * np.cos(theta) * np.cos(phi)
                     + My * np.cos(theta) * np.sin(phi)) * exp_term
        Lxy_phi = (-Mx * np.sin(phi) + My * np.cos(phi)) * exp_term

        # Integrate in the x dimension
        Ny_theta = self.integrate_func(Nxy_theta, dx=self.dx, axis=1)
        Ny_phi = self.integrate_func(Nxy_phi, dx=self.dx, axis=1)
        Ly_theta = self.integrate_func(Lxy_theta, dx=self.dx, axis=1)
        Ly_phi = self.integrate_func(Lxy_phi, dx=self.dx, axis=1)

        # Integrate in the y dimension
        self.N_theta += self.integrate_func(Ny_theta, dx=self.dx, axis=0)
        self.N_phi += self.integrate_func(Ny_phi, dx=self.dx, axis=0)
        self.L_theta += self.integrate_func(Ly_theta, dx=self.dx, axis=0)
        self.L_phi += self.integrate_func(Ly_phi, dx=self.dx, axis=0)
