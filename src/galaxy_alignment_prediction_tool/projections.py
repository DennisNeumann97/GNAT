import numpy as np

from src.galaxy_alignment_prediction_tool.multipoles import multipoles
from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum
import scipy as sp
import scipy.integrate as sp_int

class projections:
    def __init__(
        self
    ):
        """
        Class to handle projection integrals for obtaining 3D to 2D projected correlation functions
        """
        pass

    def _save_interpolation(
        self,
        x, 
        xp, 
        fp,
    ) -> np.ndarray:
        """
        Internal method to handle interpolation with out-of-bounds checking

        Args:
            x (np.ndarray): x values to interpolate to
            xp (np.ndarray): known x values
            fp (np.ndarray): known function values at xp
        Returns:
            np.ndarray: interpolated function values at x
        """
        
        if np.any(x < np.min(xp)) or np.any(x > np.max(xp)):
            raise ValueError("x values for interpolation are out of bounds of xp")
        
        return np.interp(x, xp, fp)

    def convert_multipoles_to_projected_correlation_function(
        self,
        r_list: list,
        xi_ell_list: list,
        ell_list: list,
        spin: int,
        rp_array: np.ndarray,
        pi_max: float,
        pi_min: float=1e-5,
        pi_num: int=1000,
        pi_gridding: str='logarithmic',
    ) -> np.ndarray:
        """
        Method to project a given multipole correlation function xi_{ab,ell} to a 2D correlation function
        w_{ab,ell}(rp) using the standard projection integral.

        Args:
            r_list (list): list of radial distances corresponding to each xi_ell in xi_ell_list
            xi_ell_list (list): list of multipole correlation function arrays of different order considered for the projection
            ell_list (int): multipole  list
            spin (int): spin of the correlation function (0 for gg, 2 for g+)
            rp_array (np.ndarray): array of projected radial distances rp
            pi_max (float): maximum line-of-sight separation for the projection integral, in Mpc(!!)
            pi_min (float, optional): minimum line-of-sight separation for the projection integral. Defaults to 1e-5. Necessary because of numerical issues at pi=0.
            pi_num (int, optional): number of pi points for the integral. Defaults to 1000.
            pi_gridding (str, optional): gridding type for the pi array, either 'logarithmic' or 'linear'. Defaults to 'logarithmic'.

        Returns:
            np.ndarray: projected 2D correlation function w_{ab,ell}(rp)
        """

        if pi_gridding == 'logarithmic':
            pi_array = np.geomspace(pi_min, pi_max, pi_num)
        elif pi_gridding == 'linear':
            pi_array = np.linspace(pi_min, pi_max, pi_num)
        else:
            raise ValueError("pi_gridding must be either 'logarithmic' or 'linear'")
        
        # Calculate the contribution from each multipole
        w_ell_contributions = []
        for idx, ell_val in enumerate(ell_list):
            
            # Define array we want to use for integration
            r_array = np.sqrt(rp_array[:, None]**2 + pi_array[None, :]**2)

            # Project xi_ell to the 2d array we are interested in
            xi_ell_spline = self._save_interpolation(r_array, r_list[idx], xi_ell_list[idx])
            
            # Define associated Legendre polynomial factor
            pi_over_r = pi_array[None, :] / r_array
            legendre_factor = sp.special.assoc_legendre_p(ell_val, spin, pi_over_r)[0]

            # Calculate integral over pi
            integrand = xi_ell_spline * legendre_factor
            w_ell = 2 * sp_int.simpson(integrand, pi_array, axis=1)
            w_ell_contributions.append(w_ell)

        return np.sum(w_ell_contributions, axis=0)