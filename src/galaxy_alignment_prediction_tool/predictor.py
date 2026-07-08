import numpy as np
from scipy import interpolate
import scipy.integrate as sp_int
import scipy.special as sp_special
from scipy.interpolate import RectBivariateSpline
import pyccl as ccl
from copy import copy
from math import factorial
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

        # Predefine dict with prope_type and bessel order
        self.probetype_dict = {
            'gg': 0,  # galaxy-galaxy (position-position)
            'g+': 2,  # galaxy-intrinsic alignment (position-shape)
            'spin0': 0,  # spin0 component from intrinsic-intrinsic alignment (shape-shape)
            'spin4': 4,  # spin4 component from intrinsic-intrinsic alignment (shape-shape)
        }

    def initialise_power_spectra(
        self,
        k_input: np.ndarray,
        pk_input: np.ndarray,
        probetype_order: list[str],
        interpolation_type: str='linear',
    ) -> None:
        """
        Initialise splined power spectra for each redshift in redshift_list

        Args:
            k_input (np.ndarray): shape=(len(k_input), len(pk_estimator), len(redshift_list)) wavenumber input array in 1/Mpc
            pk_input (np.ndarray): shape=(len(k_input), len(pk_estimator), len(redshift_list)) power spectrum input array in (Mpc)^3
            probetype_order (list[str]): ordered list of probe type strings corresponding to each power spectrum type.
                Each entry must be a key in self.probetype_dict (e.g. 'gg', 'g+', 'spin0', 'spin4').
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
        self.probetype_order = probetype_order

        # Save ells for multipole expansions
        self.ells = [0, 2, 4]

    def initialise_all_power_spectra_from_ccl(
        self,
        k_bounds: list[float]=[1e-5, 5e2],
        n_k: int = 2000,
        interpolation_type: str='linear',
    ) -> None:
        """
        Wrapper around initialise_power_spectra to initialise power spectra for galaxy position-position, position-shape,
        and shape-shape (spin0 and spin4 component seperately) using CCL.
        Args:
            k_bounds (list[float]): min and max wavenumber in 1/Mpc
            n_k (int): number of points to sample in k
            interpolation_type (str): type of interpolation to use ('linear', 'cubic', etc.)

        Returns:
            None
        """

        # Predefine arrays
        n_spectra = 4
        k_input = np.logspace(np.log10(k_bounds[0]), np.log10(k_bounds[1]), n_k)
        pk_input = np.zeros((n_k, n_spectra, len(self.redshift_list)))  # for Pk_g+, Pk_gg, Pk_spin0, Pk_spin4

        # Iterate over redshifts to compute Pk_mm -> Pk_g+, Pk_gg, Pk_spin0, Pk_spin4
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

            pk_spin0 = self.psHandler.convert_pk_matter_to_pk_plusplus(
                pk_matter=pk_matter,
                cosmo=self.cosmology,
                redshift=z
            )

            pk_spin4 = copy(pk_spin0)

            # Store in input array
            pk_input[:, 0, i_redshift] = pk_gplus
            pk_input[:, 1, i_redshift] = pk_gg
            pk_input[:, 2, i_redshift] = pk_spin0
            pk_input[:, 3, i_redshift] = pk_spin4


        k_input = np.tile(k_input, (len(self.redshift_list), n_spectra, 1)).T
        self.initialise_power_spectra(
            k_input=k_input,
            pk_input=pk_input,
            probetype_order=['g+', 'gg', 'spin0', 'spin4'],
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
        rp_min: float | None=None,
        mu_support: np.ndarray=np.linspace(0,1,300),
        **interp1d_kwargs,
    ) -> None:
        """
        Method to calculate multipole and projection estimators for each redshift and power spectrum type. 
        The IA and galaxy bias amplitudes are assumed to be unity and can be multiplied later.

        If rp_min is provided, wedged multipoles are additionally computed. These integrate the
        anisotropic correlation function xi(r, mu) = sum_ell alpha_ell xi_ell(r) P_ell^m(mu) over the
        mu range that corresponds to a minimum perpendicular separation rp >= rp_min at each r. Because
        rp = r * sqrt(1 - mu**2), the cut maps to |mu| <= mu_max(r) = sqrt(1 - (rp_min / r)**2), and the
        wedge is set to zero for r < rp_min. This mimics the common rp cut used in projected correlation
        function measurements to suppress non-linear (small-rp) scales.

        Args:
            rp_support (np.ndarray, optional): radial separation support array in Mpc. Defaults to np.geomspace(0.01, 200, 1000).
            beta_shape (float, optional): RSD parameter for the shape sample galaxies. Defaults to 0, i.e., no RSDs.
            beta_density (float, optional): RSD parameter for the density sample galaxies. Defaults to 0, i.e., no RSDs.
            pi_max (float, optional): maximum line-of-sight separation in Mpc. Defaults to 100.
            pi_min (float, optional): minimum line-of-sight separation in Mpc. Defaults to 1e-6.
            pi_num (int, optional): number of line-of-sight points. Defaults to 1000.
            pi_gridding (str, optional): gridding type for line-of-sight points. Defaults to 'logarithmic'.
            rp_min (float | None, optional): minimum perpendicular separation in Mpc for the wedged
                multipole cut. If None (default), wedged multipoles are not computed.
            mu_num (int, optional): number of mu points used in the wedge angular integral. Defaults to 1000.
            **interp1d_kwargs: additional keyword arguments for scipy.interpolate.interp1d when creating splines.
                I recommend leaving "bound_errors=True" to avoid unintended extrapolation effects.
        """

        self.logger.info("Calculating estimators for each redshift and power spectrum type")

        # Predefining lists to store results
        self.multipole_splines = []
        self.correlation_splines = []
        self.projection_splines = []
        self.multipole_wedged_splines = [] if rp_min is not None else None
        self.wedge_ell_list = None  # multipole orders present in the wedged splines

        # Iterate over redshifts
        for i_redshift in tqdm(range(self.n_redshifts)):
            pk_splines_z = self.pk_splines_list[i_redshift]
            multipole_splines_z = []
            projection_splines_z = []
            correlation_splines_z = []
            multipole_wedged_splines_z = [] if rp_min is not None else None

            # Iterate over power spectrum types
            for i_pk_type in range(self.n_pk_types):
                pk_spline = pk_splines_z[i_pk_type]
                probe_type = self.probetype_order[i_pk_type]
                spin = self.probetype_dict[probe_type]

                # Calculate multipoles
                # -------------------------------
                r_xi_list, xi_ell_list, ell_list = self.multipolesHandler.return_all_multipoles_from_power_spectrum(
                    power_spectrum_spline=pk_spline,
                    probe_type=probe_type,
                    beta_shape=beta_shape,
                    beta_density=beta_density,
                )

                self.logger.debug(f"Redshift {self.redshift_list[i_redshift]}, pk_type {i_pk_type}: Calculated multipoles for ell: {ell_list}")
                xi_ell_interp_dict = {ell: interpolate.interp1d(r_xi_list[i], xi_ell_list[i], **interp1d_kwargs) for i, ell in enumerate(ell_list)}

                self.logger.debug(f"xi_ell_total shape before projection to r_support: {xi_ell_list[0].shape}")
                multipole_splines_z.append(xi_ell_interp_dict)
                # -------------------------------

                # Calculate 3d correlation function
                # -------------------------------
                xi_2d_array = np.zeros(shape=(len(r_xi_list[0]), len(mu_support)))
                for ell_idx, ell in enumerate(ell_list):
                    assert np.allclose(r_xi_list[ell_idx], r_xi_list[0]), "r support not identical for different ell contributions for multipoles. Welp, that sucks..."
                    P_ell_mu = sp_special.assoc_legendre_p(ell, spin, mu_support)[0]  # shape (len(mu_support),)
                    xi_2d_array += xi_ell_list[ell_idx][:, None] * P_ell_mu[None, :]  # shape (len(r_support), len(mu_support))

                # Create 2D spline
                xi_2d_spline = RectBivariateSpline(r_xi_list[0], mu_support, xi_2d_array)
                correlation_splines_z.append(xi_2d_spline)
                # -------------------------------

                # Create wedged multipoles if rp_min is parsed
                # -------------------------------
                if rp_min is not None:
                    multipole_wedged_splines_z.append(
                        self._calculate_wedged_multipoles(
                            correlation_spline=xi_2d_spline,
                            ell_list=ell_list,
                            rp_min=rp_min,
                            spin=spin,
                            r_support=r_xi_list[0],
                            mu_support=mu_support,
                        )
                    )
                # -------------------------------

                # Calculate projections
                # -------------------------------
                w_array = self.projectionsHandler.convert_multipoles_to_projected_correlation_function(
                    r_list = r_xi_list,
                    xi_ell_list = xi_ell_list,
                    ell_list = ell_list,
                    spin=spin,
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
            self.correlation_splines.append(correlation_splines_z)
            self.multipole_wedged_splines.append(multipole_wedged_splines_z) if rp_min is not None else None

        self.logger.info("Done calculating estimators. Results stored in self.multipole_splines, self.projection_splines, and self.correlation_splines")

    def _calculate_wedged_multipoles(
        self,
        correlation_spline: RectBivariateSpline,
        ell_list: list,
        rp_min: float,
        spin: int,
        r_support: np.ndarray,
        mu_support: np.ndarray,
    ) -> dict:

        # Get mu_max_arr from r_support
        # will be NaN for r_support<rp_min, but second condition below takes care of that
        mu_r_max_arr = np.sqrt(1-rp_min**2/r_support**2)

        # Preload correlation function
        xi_2d = correlation_spline(r_support, mu_support)
        
        # Set to zero where mu smaller than mu_max, and r smaller than rp_min
        xi_2d[mu_support[None, :] > mu_r_max_arr[:, None]] = 0
        xi_2d[r_support < rp_min, :] = 0

        multipole_wedged_splines_ell = {}
        for ell in ell_list:
            if ell < spin:
                prefactor = 0
            else:
                prefactor = (2*ell+1)/2*sp_special.factorial(ell-spin)/sp_special.factorial(ell+spin)

            # Integrate along mu
            legendre = sp_special.assoc_legendre_p(ell, spin, mu_support)[0]
            integrand = xi_2d * legendre[None, :]
            xi_ell = np.trapezoid(integrand, mu_support, axis=1)

            # Factor two because even integration from 0 to 1
            multipole_wedged_splines_ell[ell] = (
                interpolate.interp1d(r_support, prefactor*2*xi_ell) 
            )
        
        return multipole_wedged_splines_ell

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
    ) -> list[list[dict[int, interpolate.interp1d]]]:
        """
        Return the calculated multipole and projection estimators

        Returns:
            list[list[dict[int, interpolate.interp1d]]]: multipole_splines for each redshift and measurement
                type, keyed by multipole order ell
        Raises:
            ValueError: if the estimators have not been calculated
        """

        if self.multipole_splines is None:
            raise ValueError("Estimators have not been calculated yet. Please run calculate_estimators() first.")

        return self.multipole_splines

    def return_multipole_wedged_splines(
        self,
    ) -> list[list[dict[int, interpolate.interp1d]]]:
        """
        Return the calculated multipole and projection estimators

        Returns:
            list[list[dict[int, interpolate.interp1d]]]: wedged multipole_splines for each redshift and
                measurement type, keyed by multipole order ell
        Raises:
            ValueError: if the estimators have not been calculated
        """

        if self.multipole_wedged_splines is None:
            raise ValueError("Estimators have not been calculated yet. Please run calculate_estimators() first.")

        return self.multipole_wedged_splines

    def bin_average_w(
        self,
        rp_edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Post-processing method to bin-average the projected correlation function
        w(rp) over rp bins, matching the area-weighted averaging of the measurement
        pipeline (RR weight: rp drp).

        For each bin [a, b]:
            w_bin = (2 / (b**2 - a**2)) * int_a^b w(rp) * rp drp

        Args:
            rp_edges (np.ndarray): bin edges for rp in Mpc, length N+1

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - rp_centers (np.ndarray): arithmetic midpoints of bins, length N
                - w_binned (np.ndarray): bin-averaged w values, shape (N, n_pk, n_redshifts)

        Raises:
            ValueError: if projection splines have not been calculated
            ValueError: if rp_edges extend beyond spline domain
        """

        if self.projection_splines is None:
            raise ValueError("Projection splines have not been calculated. "
                             "Please run calculate_estimators() first.")

        rp_centers = (rp_edges[:-1] + rp_edges[1:]) / 2
        n_bins = len(rp_centers)

        w_binned = np.zeros((n_bins, self.n_pk_types, self.n_redshifts))

        for i_redshift in range(self.n_redshifts):
            for i_pk_type in range(self.n_pk_types):
                w_spline = self.projection_splines[i_redshift][i_pk_type]

                if rp_edges[0] < w_spline.x.min() or rp_edges[-1] > w_spline.x.max():
                    raise ValueError(
                        f"rp_edges [{rp_edges[0]}, {rp_edges[-1]}] extend beyond "
                        f"spline domain [{w_spline.x.min()}, {w_spline.x.max()}] "
                        f"for redshift {self.redshift_list[i_redshift]}, pk_type {i_pk_type}"
                    )

                for i_bin in range(n_bins):
                    a, b = rp_edges[i_bin], rp_edges[i_bin + 1]
                    integral, _ = sp_int.quad(
                        lambda x: w_spline(x) * x,
                        a, b,
                        limit=200,
                    )
                    w_binned[i_bin, i_pk_type, i_redshift] = 2 * integral / (b**2 - a**2)

        return rp_centers, w_binned

    def bin_average_xi(
        self,
        r_edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Post-processing method to bin-average the multipole correlation function
        xi(r) over r bins, matching the volume-weighted averaging of the measurement
        pipeline (RR weight: r^2 dr).

        For each bin [a, b]:
            xi_bin = (3 / (b**3 - a**3)) * int_a^b xi(r) * r^2 dr

        Args:
            r_edges (np.ndarray): bin edges for r in Mpc, length N+1

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - r_centers (np.ndarray): arithmetic midpoints of bins, length N
                - xi_binned (np.ndarray): bin-averaged xi values per multipole order, shape
                  (N, n_pk, n_redshifts, n_ell), with the ell axis ordered as in self.ells

        Raises:
            ValueError: if multipole splines have not been calculated
            ValueError: if r_edges extend beyond spline domain
        """

        if self.multipole_splines is None:
            raise ValueError("Multipole splines have not been calculated. "
                             "Please run calculate_estimators() first.")

        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        n_bins = len(r_centers)

        xi_binned = np.zeros((n_bins, self.n_pk_types, self.n_redshifts, len(self.ells)))

        for i_redshift in range(self.n_redshifts):
            for i_pk_type in range(self.n_pk_types):
                for i_ell, ell in enumerate(self.ells):
                    xi_spline = self.multipole_splines[i_redshift][i_pk_type][ell]

                    if r_edges[0] < xi_spline.x.min() or r_edges[-1] > xi_spline.x.max():
                        raise ValueError(
                            f"r_edges [{r_edges[0]}, {r_edges[-1]}] extend beyond "
                            f"spline domain [{xi_spline.x.min()}, {xi_spline.x.max()}] "
                            f"for redshift {self.redshift_list[i_redshift]}, pk_type {i_pk_type}, ell {ell}"
                        )

                    for i_bin in range(n_bins):
                        a, b = r_edges[i_bin], r_edges[i_bin + 1]
                        integral, _ = sp_int.quad(
                            lambda x: xi_spline(x) * x**2,
                            a, b,
                            limit=200,
                        )
                        xi_binned[i_bin, i_pk_type, i_redshift, i_ell] = 3 * integral / (b**3 - a**3)

        return r_centers, xi_binned

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
        xi_array = np.zeros((n_r, self.n_pk_types, self.n_redshifts, len(self.ells)))*np.nan

        for i_redshift in range(self.n_redshifts):
            for i_pk_type in range(self.n_pk_types):
                w_spline = self.projection_splines[i_redshift][i_pk_type]
                w_array[:, i_pk_type, i_redshift] = w_spline(rp_support)

                for i_ell, ell in enumerate(self.ells):
                    xi_spline = self.multipole_splines[i_redshift][i_pk_type][ell]
                    xi_array[:, i_pk_type, i_redshift, i_ell] = xi_spline(r_support)

        return w_array, xi_array