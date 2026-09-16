import numpy as np
import mcfit
import scipy as sp

class multipoles:
    def __init__(
        self
    ):
        """
        Class to calculate the multipole estimator for a given power spectrum using the 
        FFTlog algorithm from mcfit (https://github.com/eelregit/mcfit)
        """
        pass
    
    def return_multipole_prefactors_for_gplus(
        self,
        beta_density: float=0.,
    ) -> list[float]:
        """
        Returns the geometric prefactors for the individual g+ multipoles, taken from
        https://arxiv.org/abs/2307.02545. TODO: Find out exactly how the Kaiser factors changed these.

        Args:
            beta_density (float, optional): RSD parameter for the density sample galaxies. Defaults to 0, i.e., no RSDs.

        Returns:
            list[float]: prefactors [alpha_2gplus, alpha_4gplus]
        """

        alpha_2gplus = 1/3 * (1 + 1/7 * beta_density)
        alpha_4gplus = 2/105 * beta_density
        
        return [0, alpha_2gplus, alpha_4gplus]
    
    def return_multipole_prefactors_for_gg(
        self,
        beta_shape: float=0.,
        beta_density: float=0.,
    ) -> list[float]:
        """
        Returns the geometric prefactors for the individual gg multipoles, taken from
        https://arxiv.org/abs/2307.02545. TODO: Find out exactly how the Kaiser factors changed these.

        Args:
            beta_shape (float, optional): RSD parameter for the shape sample galaxies. Defaults to 0, i.e., no RSDs.
            beta_density (float, optional): RSD parameter for the density sample galaxies. Defaults to 0, i.e., no RSDs.
        Returns:
            list[float]: prefactors [alpha_0gg, alpha_2gg, alpha_4gg]
        """

        alpha_0gg = 1 + 1/3 * (beta_shape + beta_density) + 1/5 * beta_shape * beta_density
        alpha_2gg = 2/3 * (beta_shape + beta_density) + 4/7 * beta_shape * beta_density
        alpha_4gg = 8/35 * beta_shape * beta_density
        
        return [alpha_0gg, alpha_2gg, alpha_4gg]
    
    def return_multipole_prefactors_for_plusplus(
        self,
        mode: str,
    ) -> list[float]:
        """
        Returns the geometric prefactors for the single ++ (or xx) multipole,

        Args:
            mode (str): Parameter defining which spin mode to return, 'spin4', or 'spin0'
        
        Returns:
            list[float]: prefactors [alpha_4plusplus]
        """

        if mode == 'spin4':
            return [0, 0, 1/105]
        elif mode == 'spin0':
            return [8/15, -16/21, 8/35]     
        else:
            raise ValueError("Unknown input mode, use either 'spin4' or 'spin0'.")

    def _return_unscaled_multipole_from_power_spectrum(
        self,
        power_spectrum_spline: sp.interpolate.interp1d,
        ell: int,
        k_bounds: tuple | None=None,
        k_num: int=1000,
    ):
        """Method to calculate the correlation function multipole signal xi_{ab,ell}

        Args:
            power_spectrum_spline (sp.interpolate.interp1d): power spectrum spline obtained from the powerSpectrum class
            ell (int): multipole order
            k_bounds (tuple, optional): bounds for the wavenumber k. Defaults to input k range.
            k_num (int, optional): number of k points. Defaults to 1000.

        Returns:
            tuple:
                - r_mcfit (np.ndarray): radial distances at which the multipole is evaluated
                - xi_ell_mcfit (np.ndarray): multipole correlation function values
        """
        
        if not k_bounds:
            k_bounds = (power_spectrum_spline.x.min(), power_spectrum_spline.x.max())
        k_support = np.geomspace(k_bounds[0], k_bounds[1], k_num)
        power_spectrum = power_spectrum_spline(k_support)

        r_mcfit, xi_ell_mcfit = mcfit.P2xi(k_support, l=ell)(power_spectrum)

        return r_mcfit, xi_ell_mcfit
    
    def return_all_multipoles_from_power_spectrum(
        self,
        power_spectrum_spline: sp.interpolate.interp1d,
        probe_type: str='g+',
        beta_shape: float=0.,
        beta_density: float=0.,
        k_bounds: tuple | None=None,
        k_num: int=1000,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
        """Method to calculate all correlation function multipole signals xi_{ab}^{ell,spin}, for ell in [2,4] for g+ and ell in [0,2,4] for gg.

        Args:
            power_spectrum_spline (sp.interpolate.interp1d): power spectrum spline obtained from the powerSpectrum class
            probe_type (str, optional): type of probe, either 'g+', 'gg'. Defaults to 'g+'. Also determines the spin of the field (0 for gg, 2 for g+).
            beta_shape (float, optional): RSD parameter for the shape sample galaxies. Defaults to 0, i.e., no RSDs, only applicable for 'gg'.
            beta_density (float, optional): RSD parameter for the density sample galaxies. Defaults to 0, i.e., no RSDs, only applicable for 'gg' and 'g+'.
            k_bounds (tuple, optional): bounds for the wavenumber k. Defaults to input k range.
            k_num (int, optional): number of k points. Defaults to 1000.
        Returns:
            tuple:
                - r_list (list[np.ndarray]): list of radial distances arrays for each multipole in Mpc.
                - xi_ell_list (list[np.ndarray]): list of multipole correlation function arrays for each multipole ell
                - ell_list (list[int]): list of multipole orders considered
        """

        # Get geometric factor for spin and ell
        if probe_type == 'g+':
            ell_list = [0,2,4]
            alpha_ells = self.return_multipole_prefactors_for_gplus(beta_density=beta_density)
        elif probe_type == 'gg':
            ell_list = [0,2,4]
            alpha_ells = self.return_multipole_prefactors_for_gg(beta_shape=beta_shape, beta_density=beta_density)
        elif probe_type in ['spin0', 'spin4']:
            ell_list = [0,2,4]
            alpha_ells = self.return_multipole_prefactors_for_plusplus(mode=probe_type)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}. Use 'g+' or 'gg'.")


        r_mcfit_list = []
        xi_ell_mcfit_list = []

        for ell in ell_list:
            r_mcfit, xi_ell_mcfit = self._return_unscaled_multipole_from_power_spectrum(
                power_spectrum_spline=power_spectrum_spline,
                ell=ell,
                k_bounds=k_bounds,
                k_num=k_num,
            )

            xi_ell_mcfit_scaled = alpha_ells[ell_list.index(ell)] * xi_ell_mcfit

            r_mcfit_list.append(r_mcfit)
            xi_ell_mcfit_list.append(xi_ell_mcfit_scaled)

        return r_mcfit_list, xi_ell_mcfit_list, ell_list