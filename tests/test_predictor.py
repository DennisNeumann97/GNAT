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

from src.galaxy_alignment_prediction_tool.predictor import galaxyAlignmentPredictor
class TestgalaxyAlignmentPredictor(object):
    """
    @class TestgalaxyAlignmentPredictor

    Integration tests for the final predictor
    """

    # Initialise cosmology and redshift list for tests
    cosmo_dict = {
        'Omega_c': 0.2603, 
        'Omega_b': 0.0486, 
        'h': 0.6774, 
        'sigma8': 0.8159, 
        'n_s': 0.9667
    }
    ccl_cosmo = ccl.Cosmology(**cosmo_dict)
    redshift_list = np.arange(0., 2.1, 0.25)

    # Define support
    rp_array = np.geomspace(0.1, 200, 200)/ccl_cosmo['h'] # Mpc
    r_array = rp_array

    pi_max = 102.49062/ccl_cosmo['h']  # Mpc

    # Load in results from period integrator code for comparison
    data_dir = Path(__file__).parent / "test_data"
    rp_gp, wgp_true = np.loadtxt(data_dir / "w_gp_z0.0.txt", unpack=True)
    r_gp, xi_gp22_true = np.loadtxt(data_dir / "xi_gp22_z0.0.txt", unpack=True)
    
    # Remove h dependencies
    rp_gp /= ccl_cosmo['h']
    r_gp /= ccl_cosmo['h']
    wgp_true /= ccl_cosmo['h']

    def test_predicted_multipoles_projection(self):
        predictor = galaxyAlignmentPredictor(
            cosmology=self.ccl_cosmo,
            redshift_list=self.redshift_list,
        )

        predictor.initialise_gp_gg_power_spectra_from_ccl(
            k_bounds=[1e-6, 1e3],
            n_k=5000,
        )

        predictor.calculate_estimators(
            rp_support=self.rp_array,
            beta_shape=0,
            beta_density=0,
            pi_max=self.pi_max,
        )

        projection_splines = predictor.return_projection_splines()
        multipole_splines = predictor.return_multipole_splines()

        w_gp_spline = projection_splines[0][0] # first redshift, first estimator
        xi_gp22_spline = multipole_splines[0][0]

        w_gp_predicted = w_gp_spline(self.rp_gp)
        xi_gp22_predicted = xi_gp22_spline(self.r_gp)

        # Compare to true results
        np.testing.assert_allclose(
            w_gp_predicted, 
            self.wgp_true,
            rtol=5e-2,
        )

        np.testing.assert_allclose(
            xi_gp22_predicted, 
            self.xi_gp22_true,
            rtol=5e-2,
        )
