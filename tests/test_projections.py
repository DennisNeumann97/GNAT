import pytest
import os
import pyccl as ccl
import numpy as np

# Add root directory to path
import sys
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
root_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum
from src.galaxy_alignment_prediction_tool.multipoles import multipoles
from src.galaxy_alignment_prediction_tool.projections import projections

class Testprojections(object):
    """
    @class Testprojections

    Unit tests for the projections handler class
    """
    projectionHandler = projections()
    multipoleHandler = multipoles()
    powerSpectrumHandler = powerSpectrum()

    # Create power spectrum for testing
    # ------------------------------
    ccl_cosmology = ccl.Cosmology(
        Omega_c=0.2603, 
        Omega_b=0.0486, 
        h=0.6774, 
        sigma8=0.8159, 
        n_s=0.9667
    )

    pi_max = 50/ccl_cosmology['h']  # Mpc

    k_array = np.geomspace(1e-6, 1e3, 1000)  # 1/Mpc
    redshift = 0.3

    # Define power spectrum
    pk_matter = powerSpectrumHandler.ccl_power_spectrum(
        k_input=k_array,
        cosmo=ccl_cosmology,
        redshift=redshift,
        pk_type='nonlinear_matter'
    )
    pk_gp = powerSpectrumHandler.convert_pk_matter_to_pk_gplus(
        pk_matter=pk_matter,
        cosmo=ccl_cosmology,
        redshift=redshift
    )
    pk_gp_spline = powerSpectrumHandler.power_spectrum_spline(
        k_input=k_array,
        pk_input=pk_gp,
        interpolation_type='linear'
    )

    pk_gg = powerSpectrumHandler.convert_pk_matter_to_pk_gg(
        pk_matter=pk_matter,
        cosmo=ccl_cosmology,
        redshift=redshift,
    )

    pk_gg_spline = powerSpectrumHandler.power_spectrum_spline(
        k_input=k_array,
        pk_input=pk_gg,
        interpolation_type='linear'
    ) 
    # ------------------------------

    # Convert to multipoles
    r_xigp_list, xigp_list, ellgp_list = multipoleHandler.return_all_multipoles_from_power_spectrum(
        power_spectrum_spline = pk_gp_spline,
        probe_type='g+',
        beta_shape=0.,
        beta_density=0.,
        k_bounds=(1e-5, 5e2),
        k_num=1000,
    )

    r_xigg_list, xigg_list, ellgg_list = multipoleHandler.return_all_multipoles_from_power_spectrum(
        power_spectrum_spline = pk_gg_spline,
        probe_type='gg',
        beta_shape=0.,
        beta_density=0.,
        k_bounds=(1e-5, 5e2),
        k_num=1000,
    )

    def test_save_interpolation(self):
        x = np.array([0.05, 0.2, 0.3, 0.4, 0.6])
        xp = np.array([0.1, 0.3, 0.5])
        fp = np.array([1.0, 3.0, 5.0])

        with pytest.raises(ValueError):
            spline = self.projectionHandler._save_interpolation(
                x=x,
                xp=xp,
                fp=fp
            )

    def test_convert_multipoles_to_projected_correlation_function(self):
        rp_array = np.linspace(0.1, 50, 10)

        # Test for g+
        w_gp = self.projectionHandler.convert_multipoles_to_projected_correlation_function(
            r_list=self.r_xigp_list,
            xi_ell_list=self.xigp_list,
            ell_list=self.ellgp_list,
            spin=2,
            rp_array=rp_array,
            pi_max=self.pi_max,
        )

        assert w_gp.shape == (len(rp_array),), "Output shape mismatch for g+ projection"

        # Test for gg
        w_gg = self.projectionHandler.convert_multipoles_to_projected_correlation_function(
            r_list=self.r_xigg_list,
            xi_ell_list=self.xigg_list,
            ell_list=self.ellgg_list,
            spin=0,
            rp_array=rp_array,
            pi_max=self.pi_max,
        )

        assert w_gg.shape == (len(rp_array),), "Output shape mismatch for gg projection"

        # Test hyperparameters
        # ----------------------------------------
        # Pi min test
        default_pi_min = 1e-5
        test_pi_min_list = [1e-8, 1e-7, 1e-6, 1e-4, 1e-3]
        for test_pi_min in test_pi_min_list:
            w_gg_pimin_test = self.projectionHandler.convert_multipoles_to_projected_correlation_function(
                r_list=self.r_xigg_list,
                xi_ell_list=self.xigg_list,
                ell_list=self.ellgg_list,
                spin=0,
                rp_array=rp_array,
                pi_max=self.pi_max,
                pi_min=test_pi_min,
            )

            np.testing.assert_allclose(
                w_gg, 
                w_gg_pimin_test, 
                rtol=1e-2,
                err_msg=f"w_gg result changed significantly when changing pi_min from {default_pi_min} to {test_pi_min}"
            )

            w_gp_pimin_test = self.projectionHandler.convert_multipoles_to_projected_correlation_function(
                r_list=self.r_xigp_list,
                xi_ell_list=self.xigp_list,
                ell_list=self.ellgp_list,
                spin=2,
                rp_array=rp_array,
                pi_max=self.pi_max,
                pi_min=test_pi_min,
            )

            np.testing.assert_allclose(
                w_gp, 
                w_gp_pimin_test, 
                rtol=1e-2,
                err_msg=f"w_gp result changed significantly when changing pi_min from {default_pi_min} to {test_pi_min}"
            )

        # Pi gridding test
        pi_gridding_types = ['linear', 'logarithmic']
        for pi_gridding in pi_gridding_types:
            w_gg_pigrid_test = self.projectionHandler.convert_multipoles_to_projected_correlation_function(
                r_list=self.r_xigg_list,
                xi_ell_list=self.xigg_list,
                ell_list=self.ellgg_list,
                spin=0,
                rp_array=rp_array,
                pi_max=self.pi_max,
                pi_gridding=pi_gridding,
            )

            np.testing.assert_allclose(
                w_gg, 
                w_gg_pigrid_test, 
                rtol=1e-2,
                err_msg=f"w_gg result changed significantly when changing pi_gridding to {pi_gridding}"
            )

            w_gp_pigrid_test = self.projectionHandler.convert_multipoles_to_projected_correlation_function(
                r_list=self.r_xigp_list,
                xi_ell_list=self.xigp_list,
                ell_list=self.ellgp_list,
                spin=2,
                rp_array=rp_array,
                pi_max=self.pi_max,
                pi_gridding=pi_gridding,
            )

            np.testing.assert_allclose(
                w_gp, 
                w_gp_pigrid_test, 
                rtol=1e-2,
                err_msg=f"w_gp result changed significantly when changing pi_gridding to {pi_gridding}"
            )
        # ----------------------------------------
        
        