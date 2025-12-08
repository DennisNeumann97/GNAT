import os
import numpy as np
from scipy import interpolate
import pyccl as ccl

class powerSpectrum:
    def __init__(
        self,
    ):
        """
        Class that handles input power spectra by splining, interpolating, etc...
        TODO: Fill this out once there is more inside
        """

    def power_spectrum_spline(
        self,
        k_input: np.ndarray,
        pk_input: np.ndarray,
        interpolation_type: str='linear'
    ) -> interpolate.interp1d:
        """
        Wrapper around scipy.interpolation used for resampling
        an input power spectrum to an arbitrary k

        Args:
            k_input (np.ndarray): wavenumber input array in 1/Mpc
            pk_input (np.ndarray): power spectrum input array in (Mpc)^3
        """
        
        pk_spline = interpolate.interp1d(k_input, pk_input, kind=interpolation_type, bounds_error=True)

        return pk_spline
    
    def ccl_power_spectrum(
        self,
        cosmo: ccl.Cosmology,
        k_input: np.ndarray,
        redshift: float,
        pk_type: str='nonlinear_matter'
    ) -> np.ndarray:
        """
        Wrapper around CCL power spectrum calculator

        Args:
            cosmo (ccl.Cosmology): CCL cosmology object
            k_input (np.ndarray): wavenumber input array in 1/Mpc
            a_input (np.ndarray): scale factor input array
            pk_type (str): type of power spectrum to compute ('matter', 'linear_matter', etc.)

        Returns:
            np.ndarray: computed power spectrum values
        """
        
        scale_factor = 1/(1+redshift)

        if pk_type == 'linear_matter':
            pk_output = ccl.linear_matter_power(cosmo, k_input, scale_factor)
        elif pk_type == 'nonlinear_matter':
            pk_output = ccl.nonlin_matter_power(cosmo, k_input, scale_factor)
        else:
            raise ValueError(f"Unknown power spectrum type: {pk_type}")

        return pk_output
    
    def convert_pk_matter_to_pk_gplus(
        self,
        pk_matter: np.ndarray,
        cosmo: ccl.Cosmology,
        redshift: float,
        C1rhocrit: float=0.0134,
    ) -> np.ndarray:
        """Outputs Pk_g+ from an input Pk_mm, assuming alignment amplitude and galaxy bias is unity.

        Args:
            pk_matter (np.ndarray): matter power spectrum amplitude in 1/Mpc^3
            cosmo (ccl.Cosmology): ccl.cosmology instance
            redshift (float): redshift of the Pk_mm

        Returns:
            np.ndarray: Pk_g+, galaxy-shape 3d power spectrum
        """
    
        scale_factor = 1/(1+redshift)
        prefactor = - C1rhocrit*cosmo['Omega_m']/ccl.growth_factor(cosmo, scale_factor)

        return pk_matter * prefactor
    
    def convert_pk_matter_to_pk_gg(
        self,
        pk_matter: np.ndarray,
        cosmo: ccl.Cosmology,
        redshift: float,
    ) -> np.ndarray:
        """Outputs Pk_gg from an input Pk_mm, assuming galaxy bias is unity. Essentially a passthrough.

        Args:
            pk_matter (np.ndarray): matter power spectrum amplitude in 1/Mpc^3
            cosmo (ccl.Cosmology): ccl.cosmology instance
            redshift (float): redshift of the Pk_mm

        Returns:
            np.ndarray: Pk_gg, galaxy-galaxy 3d power spectrum
        """

        return pk_matter
    
    def convert_pk_matter_to_pk_plusplus(
        self,
        pk_matter: np.ndarray,
        cosmo: ccl.Cosmology,
        redshift: float,
        C1rhocrit: float=0.0134,
    ) -> np.ndarray:
        """Outputs Pk_++, shape-shape 3d power spectrum from an input Pk_mm, assuming alignment amplitude is unity.

        Args:
            pk_matter (np.ndarray): matter power spectrum amplitude in 1/Mpc^3
            cosmo (ccl.Cosmology): ccl.cosmology instance
            redshift (float): redshift of the Pk_mm 

        Returns:
            np.ndarray: Pk_++, shape-shape 3d power spectrum
        """
            
        scale_factor = 1/(1+redshift)
        prefactor = (C1rhocrit*cosmo['Omega_m']/ccl.growth_factor(cosmo, scale_factor))**2

        return pk_matter * prefactor