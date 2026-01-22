import pytest
import os
import pyccl as ccl
import numpy as np
from pathlib import Path

# Add root directory to path
import sys
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
root_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum
class TestpowerSpectrum(object):
    """
    @class TestpowerSpectrum

    Unit tests for the power spectrum handler class
    """

    test_cosmo_dict = {
        'Omega_c': 0.2603, 
        'Omega_b': 0.0486, 
        'h': 0.6774, 
        'sigma8': 0.8159, 
        'n_s': 0.9667
    }
    k_array = np.geomspace(1e-6, 1e3, 1000)
    test_cosmo = ccl.Cosmology(**test_cosmo_dict)

    psHandler = powerSpectrum()

    def test_ccl_power_spectrum(self):

        redshift = 0.5

        pk_ccl_nonlin = self.psHandler.ccl_power_spectrum(
            k_input=self.k_array,
            cosmo=self.test_cosmo,
            redshift=redshift,
            pk_type='nonlinear_matter'
        )
        pk_ccl_lin = self.psHandler.ccl_power_spectrum(
            k_input=self.k_array,
            cosmo=self.test_cosmo,
            redshift=redshift,
            pk_type='linear_matter'
        )

        # Expected values from CCL directly
        pk_expected_nonlin = ccl.nonlin_matter_power(self.test_cosmo, self.k_array, 1/(1+redshift))
        pk_expected_lin = ccl.linear_matter_power(self.test_cosmo, self.k_array, 1/(1+redshift))

        np.testing.assert_allclose(pk_ccl_nonlin, pk_expected_nonlin, rtol=1e-5)
        np.testing.assert_allclose(pk_ccl_lin, pk_expected_lin, rtol=1e-5)

    def test_power_spectrum_spline(self):
        k_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 1/Mpc
        pk_input = np.array([10.0, 20.0, 15.0, 25.0, 30.0])  # (Mpc)^3

        pk_spline = self.psHandler.power_spectrum_spline(
            k_input=k_input,
            pk_input=pk_input,
            interpolation_type='linear'
        )

        # Test interpolation at known points
        for k_val, pk_val in zip(k_input, pk_input):
            np.testing.assert_allclose(pk_spline(k_val), pk_val, rtol=1e-5)

        # Test interpolation at midpoints
        k_test = 0.15
        pk_expected = (20.0 + 10.0) / 2
        np.testing.assert_allclose(pk_spline(k_test), pk_expected, rtol=1e-5)

        # Test extrapolation raises error
        with pytest.raises(ValueError):
            pk_spline(0.05)

    def test_convert_pk_matter_to_pk_gplus(self):
        redshift = 0.5

        pk_matter = self.psHandler.ccl_power_spectrum(
            k_input=self.k_array,
            cosmo=self.test_cosmo,
            redshift=redshift,
            pk_type='nonlinear_matter'
        )

        pk_gplus = self.psHandler.convert_pk_matter_to_pk_gplus(
            pk_matter=pk_matter,
            cosmo=self.test_cosmo,
            redshift=redshift,
        )

        # Basic sanity checks
        assert pk_gplus.shape == pk_matter.shape, "Output shape mismatch"

        # Check scaling
        C1rhocrit = 5e-14 * ccl.physical_constants.RHO_CRITICAL
        expected_scaling = self.test_cosmo['Omega_m'] / ccl.growth_factor(self.test_cosmo, 1/(1+redshift))
        expected_pk_gplus = - C1rhocrit * expected_scaling * pk_matter

        np.testing.assert_allclose(pk_gplus, expected_pk_gplus, rtol=1e-5)

    def test_pk_matter_to_pk_gg(self):
        redshift = 0.5

        pk_matter = self.psHandler.ccl_power_spectrum(
            k_input=self.k_array,
            cosmo=self.test_cosmo,
            redshift=redshift,
            pk_type='nonlinear_matter'
        )

        pk_gg = self.psHandler.convert_pk_matter_to_pk_gg(
            pk_matter=pk_matter,
            cosmo=self.test_cosmo,
            redshift=redshift,
        )

        # Basic sanity checks
        assert pk_gg.shape == pk_matter.shape, "Output shape mismatch"

        # With bias == 1, pk_gg should equal pk_matter
        expected_pk_gg = pk_matter

        np.testing.assert_allclose(pk_gg, expected_pk_gg, rtol=1e-5)
