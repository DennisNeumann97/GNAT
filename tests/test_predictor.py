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
    # redshift_list = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    redshift_list = np.array([0.0, 0.5])

    # Define support
    rp_array = np.geomspace(0.1, 200, 200)/ccl_cosmo['h'] # Mpc
    r_array = rp_array

    pi_max = 50/ccl_cosmo['h']  # Mpc

    # Load in results from period integrator code for comparison
    data_dir = Path(__file__).parent / "test_data" / "predictions_from_period_integrator"

    rp_gp = []
    wgp_true = []

    rp_gg = []
    wgg_true = []

    r_gp = []
    xi_gp22_true = []

    for redshift in redshift_list:
        rp_gp_z, wgp_z = np.loadtxt(data_dir / f"w_gp_z{redshift:.1f}.txt", unpack=True)
        rp_gg_z, wgg_z = np.loadtxt(data_dir / f"w_gg_z{redshift:.1f}.txt", unpack=True)
        r_gp_z, xi_gp22_z = np.loadtxt(data_dir / f"xi_gp22_z{redshift:.1f}.txt", unpack=True)

        rp_gp.append(rp_gp_z / ccl_cosmo['h'])
        wgp_true.append(wgp_z / ccl_cosmo['h'])

        rp_gg.append(rp_gg_z / ccl_cosmo['h'])
        wgg_true.append(wgg_z / ccl_cosmo['h'])

        r_gp.append(r_gp_z / ccl_cosmo['h'])
        xi_gp22_true.append(xi_gp22_z)

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

        # Check results for all redshifts
        rtol = 5e-2
        for i, redshift in enumerate(self.redshift_list):
            w_gp_spline = projection_splines[i][0] # first estimator is w_gp
            w_gg_spline = projection_splines[i][1] # second estimator is w_gg
            xi_gp22_spline = multipole_splines[i][0]

            w_gp_predicted = w_gp_spline(self.rp_gp[i])
            w_gg_predicted = w_gg_spline(self.rp_gg[i])
            xi_gp22_predicted = xi_gp22_spline(self.r_gp[i])

            # Compare to true results
            np.testing.assert_allclose(
                w_gp_predicted, 
                self.wgp_true[i],
                rtol=rtol,
            )
            np.testing.assert_allclose(
                w_gg_predicted, 
                self.wgg_true[i],
                rtol=rtol,
            )
            np.testing.assert_allclose(
                xi_gp22_predicted, 
                self.xi_gp22_true[i],
                rtol=rtol,
            )
