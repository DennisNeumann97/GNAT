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

class Testmultipoles(object):
    """
    @class Testmultipoles

    Unit tests for the multipoles handler class
    """

    multipoleHandler = multipoles()

    # Create power spectrum for testing
    # ------------------------------
    ccl_cosmology = ccl.Cosmology(
        Omega_c=0.2603, 
        Omega_b=0.0486, 
        h=0.6774, 
        sigma8=0.8159, 
        n_s=0.9667
    )

    k_array = np.geomspace(1e-4, 1e2, 1000)  # 1/Mpc
    psHandler = powerSpectrum()
    redshift = 0.3

    # Define power spectrum
    pk_matter = psHandler.ccl_power_spectrum(
        k_input=k_array,
        cosmo=ccl_cosmology,
        redshift=redshift,
        pk_type='nonlinear_matter'
    )
    pk_gp = psHandler.convert_pk_matter_to_pk_gplus(
        pk_matter=pk_matter,
        cosmo=ccl_cosmology,
        redshift=redshift
    )
    pk_gp_spline = psHandler.power_spectrum_spline(
        k_input=k_array,
        pk_input=pk_gp,
        interpolation_type='linear'
    )
    # ------------------------------


    def test_return_multipole_prefactors_for_gplus(self):
        beta_density = 0.5

        prefactors = self.multipoleHandler.return_multipole_prefactors_for_gplus(
            beta_density=beta_density
        )

        expected_alpha_gplus = [
            0,
            1/3 * (1 + 1/7 * beta_density),
            2/105 * beta_density
        ]

        for alpha_idx in range(3):
            np.testing.assert_allclose(prefactors[alpha_idx], expected_alpha_gplus[alpha_idx], rtol=1e-5)

    def test_return_multipole_prefactors_for_gg(self):
        beta_shape = 0.4
        beta_density = 0.6

        prefactors = self.multipoleHandler.return_multipole_prefactors_for_gg(
            beta_shape=beta_shape,
            beta_density=beta_density
        )

        expected_alpha_0gg = 1 + 1/3 * (beta_shape + beta_density) + 1/5 * beta_shape * beta_density
        expected_alpha_2gg = 2/3 * (beta_shape + beta_density) + 4/7 * beta_shape * beta_density
        expected_alpha_4gg = 8/35 * beta_shape * beta_density

        np.testing.assert_allclose(prefactors[0], expected_alpha_0gg, rtol=1e-5)
        np.testing.assert_allclose(prefactors[1], expected_alpha_2gg, rtol=1e-5)
        np.testing.assert_allclose(prefactors[2], expected_alpha_4gg, rtol=1e-5)

    def test_return_all_multipoles_from_power_spectrum(self):

        # Test that k_bounds outside of pk spline range raises error
        with pytest.raises(ValueError):
            self.multipoleHandler.return_all_multipoles_from_power_spectrum(
                power_spectrum_spline=self.pk_gp_spline,
                probe_type='g+',
                beta_shape=0.,
                beta_density=0.,
                k_bounds=(1e-6, 1e2),
                k_num=1000,
            )

        # Test that output shapes are correct
        ell_list_true = [[0, 2, 4], [0, 2, 4]]
        probe_types = ['gg', 'g+']
        for idx_probe, probe_type in enumerate(probe_types):
            r_xi_list, xi_list, ell_list = self.multipoleHandler.return_all_multipoles_from_power_spectrum(
                power_spectrum_spline=self.pk_gp_spline,
                probe_type=probe_type,
                beta_shape=0.,
                beta_density=0.,
                k_bounds=(1e-4, 1e2),
                k_num=1000,
            )
            assert len(r_xi_list) == len(ell_list_true[idx_probe]), "Number of output r_xi splines mismatch"
            assert len(xi_list) == len(ell_list_true[idx_probe]), "Number of output xi splines mismatch"
            assert len(ell_list) == len(ell_list_true[idx_probe]), "Number of output ell values mismatch"
            for r_xi, xi in zip(r_xi_list, xi_list):
                assert r_xi.shape == xi.shape, "r_xi and xi shape mismatch"