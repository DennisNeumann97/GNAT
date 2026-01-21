import os
import numpy as np
from scipy import interpolate
import pyccl as ccl
from tqdm import tqdm
import logging

from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum
from src.galaxy_alignment_prediction_tool.multipoles import multipoles
from src.galaxy_alignment_prediction_tool.projections import projections

class galaxyAlignmentPredictor:
    def __init__(
        self,
        cosmology: ccl.Cosmology,
        redshift_list: np.ndarray,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        """
        Class that handles galaxy alignment predictions by transforming power spectra into observables.
        TODO: Fill this out once there is more inside
        """

        # Parse inputs
        self.logger = logger
        self.cosmology = cosmology
        self.redshift_list = redshift_list

        # Initialise custom class handlers
        self.psHandler = powerSpectrum()
        self.multipolesHandler = multipoles()
        self.projectionsHandler = projections()

        # Predefine C1 and rhocrit
        C1 = 5e-14  # in (h^2 M_sun Mpc^-3)^-1
        rhocrit = ccl.physical_constants.RHO_CRITICAL # M_sun/h (Mpc/h)^-3
        self.C1rhocrit = C1 * rhocrit

    def initialise_power_spectra(
        self,
        k_input: np.ndarray,
        pk_input: np.ndarray,
        bessel_order: np.ndarray,
        interpolation_type: str='linear',
    ) -> None:
        """
        Initialise splined power spectra for each redshift in redshift_list

        Args:
            k_input (np.ndarray): shape=(len(k_input), len(pk_estimator), len(redshift_list)) wavenumber input array in 1/Mpc
            pk_input (np.ndarray): shape=(len(k_input), len(pk_estimator), len(redshift_list)) power spectrum input array in (Mpc)^3
            bessel_order (np.ndarray): len(pk_estimator) array of Bessel function orders corresponding to each power spectrum type.
                Should be 0 for position-position, and 2 for position-shape for example
            interpolation_type (str): type of interpolation to use ('linear', 'cubic', etc.)
        """

        self.logger.debug(f"Shape of k_input: {k_input.shape}")
        self.logger.debug(f"Shape of pk_input: {pk_input.shape}")
        self.logger.debug("Should correspond to (n_k, n_pk, n_redshift")

        n_pk_type = len(k_input[0,:,0])
        n_redshift = len(k_input[0,0,:])

        self.pk_splines_list = []

        self.logger.info("Initialising power spectrum splines for each redshift")
        for i_redshift in tqdm(range(n_redshift)):

            pk_splines_z = []
            for i_pk_type in range(n_pk_type):
                pk_spline = self.psHandler.power_spectrum_spline(
                    k_input=k_input[:, i_pk_type, i_redshift],
                    pk_input=pk_input[:, i_pk_type, i_redshift],
                    interpolation_type=interpolation_type
                )
                pk_splines_z.append(pk_spline)
            
            # Append the list of splines for this redshift to the main list
            self.pk_splines_list.append(pk_splines_z)

        self.logger.info("Done. Splines stored in self.pk_splines_list")

        # Save number of input pk_types and redshifts
        self.n_pk_types = n_pk_type
        self.n_redshifts = n_redshift
        self.bessel_order = bessel_order

    def initialise_gp_gg_power_spectra_from_ccl(
        self,
        k_bounds: list[float]=[1e-5, 5e2],
        n_k: int = 2000,
        interpolation_type: str='linear',
    ) -> None:
        """
        Wrapper around initialise_power_spectra to initialise power spectra for galaxy position-position and position-shape using CCL.
        Args:
            k_bounds (list[float]): min and max wavenumber in 1/Mpc
            n_k (int): number of points to sample in k
            interpolation_type (str): type of interpolation to use ('linear', 'cubic', etc.)

        Returns:
            None
        """

        # Predefine arrays
        k_input = np.logspace(np.log10(k_bounds[0]), np.log10(k_bounds[1]), n_k)
        pk_input = np.zeros((n_k, 2, len(self.redshift_list)))  # 2 for Pk_g+, Pk_gg

        # Iterate over redshifts to compute Pk_mm, -> Pk_g+, Pk_gg
        for i_redshift, z in enumerate(self.redshift_list):
            self.logger.debug(f"Computing power spectra at redshift z={z} using CCL")

            # Compute matter power spectrum using CCL
            pk_matter = self.psHandler.ccl_power_spectrum(
                cosmo=self.cosmology,
                k_input=k_input,
                redshift=z,
                pk_type='nonlinear_matter'
            )

            # Convert to Pk_g+ and Pk_gg
            pk_gplus = self.psHandler.convert_pk_matter_to_pk_gplus(
                pk_matter=pk_matter,
                cosmo=self.cosmology,
                redshift=z,
                C1rhocrit=self.C1rhocrit,
            )

            pk_gg = self.psHandler.convert_pk_matter_to_pk_gg(
                pk_matter=pk_matter,
                cosmo=self.cosmology,
                redshift=z,
            )

            # Store in input array
            pk_input[:, 0, i_redshift] = pk_gplus
            pk_input[:, 1, i_redshift] = pk_gg

        k_input = np.tile(k_input, (len(self.redshift_list), 2, 1)).T
        self.initialise_power_spectra(
            k_input=k_input,
            pk_input=pk_input,
            bessel_order=np.array([2, 0]),
            interpolation_type=interpolation_type,
        )

    def return_pk_splines_list(self) -> list[list[interpolate.interp1d]]:
        """
        Return the list of power spectrum splines for each redshift

        Returns:
            list[list[interpolate.interp1d]]: list of power spectrum splines for each redshift
        Raises:
            ValueError: if the power spectrum splines have not been initialised
        """

        if self.pk_splines_list is None:
            raise ValueError("Power spectrum splines have not been initialised")

        return self.pk_splines_list


    def calculate_estimators(
        self,
        rp_support: np.ndarray=np.geomspace(0.01, 200, 1000),   
        beta_shape: float=0.,
        beta_density: float=0.,
        pi_max: float=100.,
        pi_min: float=1e-6,
        pi_num: int=1000,
        pi_gridding: str='logarithmic',
        **interp1d_kwargs,
    ) -> None:
        """
        Method to calculate multipole and projection estimators for each redshift and power spectrum type. 
        The IA and galaxy bias amplitudes are assumed to be unity and can be multiplied later.

        Args:
            rp_support (np.ndarray, optional): radial separation support array in Mpc. Defaults to np.geomspace(0.01, 200, 1000).
            beta_shape (float, optional): RSD parameter for the shape sample galaxies. Defaults to 0, i.e., no RSDs.
            beta_density (float, optional): RSD parameter for the density sample galaxies. Defaults to 0, i.e., no RSDs.
            pi_max (float, optional): maximum line-of-sight separation in Mpc. Defaults to 100.
            pi_min (float, optional): minimum line-of-sight separation in Mpc. Defaults to 1e-6.
            pi_num (int, optional): number of line-of-sight points. Defaults to 1000.
            pi_gridding (str, optional): gridding type for line-of-sight points. Defaults to 'logarithmic'.
            **interp1d_kwargs: additional keyword arguments for scipy.interpolate.interp1d when creating splines.
                I recommend leaving "bound_errors=True" to avoid unintended extrapolation effects.
        """

        self.logger.info("Calculating estimators for each redshift and power spectrum type")

        # Predefining lists to store results
        self.multipole_splines = []
        self.projection_splines = []

        # Iterate over redshifts
        for i_redshift in tqdm(range(self.n_redshifts)):
            pk_splines_z = self.pk_splines_list[i_redshift]
            multipole_splines_z = []
            projection_splines_z = []

            # Iterate over power spectrum types
            for i_pk_type in range(self.n_pk_types):
                pk_spline = pk_splines_z[i_pk_type]
                bessel_order = self.bessel_order[i_pk_type]

                # Calculate multipoles
                # -------------------------------
                r_xi_list, xi_ell_list, ell_list = self.multipolesHandler.return_all_multipoles_from_power_spectrum(
                    power_spectrum_spline=pk_spline,
                    probe_type='g+' if bessel_order==2 else 'gg',
                    beta_shape=beta_shape,
                    beta_density=beta_density,
                )

                self.logger.debug(f"Redshift {self.redshift_list[i_redshift]}, pk_type {i_pk_type}: Calculated multipoles for ell: {ell_list}")
                xi_ell_total = np.sum(np.array(xi_ell_list), axis=0)
                self.logger.debug(f"xi_ell_total shape before projection to r_support: {xi_ell_total.shape}")
                xi_ell_interp = interpolate.interp1d(r_xi_list[0], xi_ell_total, **interp1d_kwargs)

                multipole_splines_z.append(xi_ell_interp)
                # -------------------------------

                # Calculate projections
                # -------------------------------
                w_array = self.projectionsHandler.convert_multipoles_to_projected_correlation_function(
                    r_list = r_xi_list,
                    xi_ell_list = xi_ell_list,
                    ell_list = ell_list,
                    spin=bessel_order, # TODO: THIS MAY NEED TO CHANGE IF WE ADD ++
                    rp_array=rp_support,
                    pi_max=pi_max,
                    pi_min=pi_min,
                    pi_num=pi_num,
                    pi_gridding=pi_gridding,
                )

                w_interp = interpolate.interp1d(rp_support, w_array, **interp1d_kwargs)
                self.logger.debug(f"Redshift {self.redshift_list[i_redshift]}, pk_type {i_pk_type}: Calculated projected correlation function w(rp)")
                
                projection_splines_z.append(w_interp)
                # -------------------------------

            # Append splines for this redshift      
            self.multipole_splines.append(multipole_splines_z)
            self.projection_splines.append(projection_splines_z)


        self.logger.info("Done calculating estimators. Results stored in self.multipole_splines and self.projection_splines")

    def return_projection_splines(
        self,
    ) -> list[list[interpolate.interp1d]]:
        """
        Return the calculated multipole and projection estimators

        Returns:
            list[list[interpolate.interp1d]]: projection_splines for each redshift and measurement type
        Raises:
            ValueError: if the estimators have not been calculated
        """

        if self.projection_splines is None:
            raise ValueError("Estimators have not been calculated yet. Please run calculate_estimators() first.")

        return self.projection_splines
    
    def return_multipole_splines(
        self,
    ) -> list[list[interpolate.interp1d]]:
        """
        Return the calculated multipole and projection estimators

        Returns:
            list[list[interpolate.interp1d]]: multipole_splines for each redshift and measurement type
        Raises:
            ValueError: if the estimators have not been calculated
        """

        if self.multipole_splines is None:
            raise ValueError("Estimators have not been calculated yet. Please run calculate_estimators() first.")

        return self.multipole_splines
    
    def return_all_estimators_as_arrays(
        self,
        rp_support: np.ndarray,
        r_support: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return all projection estimators as arrays for each redshift and measurement type.

        Args:
            rp_support (np.ndarray): perpendicular separation array at which to evaluate w(rp) in Mpc
            r_support (np.ndarray): total radial separation array at which to evaluate xi(r) in Mpc
        Returns:
            tuple[np.ndarray, np.ndarray]: projection estimators w(rp) and multipole estimators xi(r) as arrays.
                Output shapes are (len(rp_support), n_pk, n_redshifts) for w(rp) and xi(r), respectively.
        Raises:
            ValueError: if the estimators have not been calculated
        """

        if self.projection_splines is None or self.multipole_splines is None:
            raise ValueError("Estimators have not been calculated yet. Please run calculate_estimators() first.")

        n_rp = len(rp_support)
        n_r = len(r_support)

        w_array = np.zeros((n_rp, self.n_pk_types, self.n_redshifts))*np.nan
        xi_array = np.zeros((n_r, self.n_pk_types, self.n_redshifts))*np.nan

        for i_redshift in range(self.n_redshifts):
            for i_pk_type in range(self.n_pk_types):
                w_spline = self.projection_splines[i_redshift][i_pk_type]
                xi_spline = self.multipole_splines[i_redshift][i_pk_type]

                w_array[:, i_pk_type, i_redshift] = w_spline(rp_support)
                xi_array[:, i_pk_type, i_redshift] = xi_spline(r_support)

        return w_array, xi_array