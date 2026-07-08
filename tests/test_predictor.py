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

        predictor.initialise_all_power_spectra_from_ccl(
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
        rtol = 2e-2
        for i, redshift in enumerate(self.redshift_list):
            w_gp_spline = projection_splines[i][0] # first estimator is w_gp
            w_gg_spline = projection_splines[i][1] # second estimator is w_gg
            xi_gp22_spline = multipole_splines[i][0][2] # g+ total multipole (pk_type 0), xi_gp22, by putting ell=2 in last extension

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

    def test_multipole_wedging(self):
        
        # Define testing rp
        rp_min = 9. # ~6 Mpc/h
        rp_min_nochange = 0.

        predictor = galaxyAlignmentPredictor(
            cosmology=self.ccl_cosmo,
            redshift_list=self.redshift_list,
        )

        predictor.initialise_all_power_spectra_from_ccl(
            k_bounds=[1e-6, 1e3],
            n_k=5000,
        )

        # Calculate unwedged multipoles
        predictor.calculate_estimators(
            rp_support=self.rp_array,
            beta_shape=0,
            beta_density=0,
            pi_max=self.pi_max,
        )

        multipole_splines = predictor.return_multipole_splines()

        # Assert that calling wedged splines without intialisation raises error


        # Calculate wedged multipoles
        predictor.calculate_estimators(
            rp_support=self.rp_array,
            beta_shape=0,
            beta_density=0,
            pi_max=self.pi_max,
            rp_min=rp_min
        )

        multipole_splines_wedged = predictor.return_multipole_wedged_splines()

        # Calculate wedged multipoles with negligible cutoff
        predictor.calculate_estimators(
            rp_support=self.rp_array,
            beta_shape=0,
            beta_density=0,
            pi_max=self.pi_max,
            rp_min=rp_min_nochange
        )

        multipole_splines_wedged_nochange = predictor.return_multipole_wedged_splines()

        # Check results for all redshifts
        rtol = 2e-3
        for i_z, _ in enumerate(self.redshift_list):
            for i_pk, probetype in enumerate(predictor.probetype_order):
                ell_dominant = predictor.probetype_dict[probetype]
                xi_unwedged = multipole_splines[i_z][i_pk][ell_dominant](self.r_array)
                xi_wedged = multipole_splines_wedged[i_z][i_pk][ell_dominant](self.r_array)
                xi_wedged_nochange = multipole_splines_wedged_nochange[i_z][i_pk][ell_dominant](self.r_array)

                assert np.allclose(xi_unwedged, xi_wedged_nochange), 'Setting rp_min=0 leads to different results than not parsing it at all'
                assert np.allclose(xi_wedged[self.r_array<rp_min], np.zeros(sum(self.r_array<rp_min))), 'Multipole signal not 0 for r<rp_min in wedged case'
                assert np.isclose(xi_unwedged[-1], xi_wedged[-1], rtol=rtol), 'Wedged and unwedged multipoles do not converge on large r'

    def test_output_dimensions(self):

        rp_min = 9.
        n_z = len(self.redshift_list)

        predictor = galaxyAlignmentPredictor(
            cosmology=self.ccl_cosmo,
            redshift_list=self.redshift_list,
        )

        predictor.initialise_all_power_spectra_from_ccl(
            k_bounds=[1e-6, 1e3],
            n_k=5000,
        )

        predictor.calculate_estimators(
            rp_support=self.rp_array,
            beta_shape=0,
            beta_density=0,
            pi_max=self.pi_max,
            rp_min=rp_min,
        )

        n_pk = predictor.n_pk_types
        ells = set(predictor.ells)

        # Projection splines: list of length n_z, each holding n_pk splines
        projection_splines = predictor.return_projection_splines()
        assert len(projection_splines) == n_z
        for splines_z in projection_splines:
            assert len(splines_z) == n_pk

        # Multipole splines: list of length n_z, each holding n_pk dicts keyed by ell
        multipole_splines = predictor.return_multipole_splines()
        assert len(multipole_splines) == n_z
        for splines_z in multipole_splines:
            assert len(splines_z) == n_pk
            for ell_dict in splines_z:
                assert set(ell_dict.keys()) == ells

        # Wedged multipole splines: same nested structure as multipole_splines
        multipole_splines_wedged = predictor.return_multipole_wedged_splines()
        assert len(multipole_splines_wedged) == n_z
        for splines_z in multipole_splines_wedged:
            assert len(splines_z) == n_pk
            for ell_dict in splines_z:
                assert set(ell_dict.keys()) == ells

        # Estimator arrays: shapes (n_support, n_pk, n_z) and (n_support, n_pk, n_z, n_ell)
        n_rp, n_r = 15, 20
        rp_support = np.geomspace(self.rp_array.min(), self.rp_array.max(), n_rp)
        r_support = np.geomspace(self.r_array.min(), self.r_array.max(), n_r)
        w_array, xi_array = predictor.return_all_estimators_as_arrays(
            rp_support=rp_support,
            r_support=r_support,
        )
        assert w_array.shape == (n_rp, n_pk, n_z)
        assert xi_array.shape == (n_r, n_pk, n_z, len(ells))
        assert not np.any(np.isnan(w_array)), 'return_all_estimators_as_arrays left unfilled (NaN) entries in w_array'
        assert not np.any(np.isnan(xi_array)), 'return_all_estimators_as_arrays left unfilled (NaN) entries in xi_array'

        # Bin-averaged w(rp): centers of length n_bins, values of shape (n_bins, n_pk, n_z)
        n_bins = 9
        rp_edges = np.geomspace(self.rp_array.min(), self.rp_array.max(), n_bins + 1)
        rp_centers, w_binned = predictor.bin_average_w(rp_edges=rp_edges)
        assert rp_centers.shape == (n_bins,)
        assert w_binned.shape == (n_bins, n_pk, n_z)

        # Bin-averaged xi(r): centers of length n_bins, values of shape (n_bins, n_pk, n_z, n_ell)
        r_edges = np.geomspace(self.r_array.min(), self.r_array.max(), n_bins + 1)
        r_centers, xi_binned = predictor.bin_average_xi(r_edges=r_edges)
        assert r_centers.shape == (n_bins,)
        assert xi_binned.shape == (n_bins, n_pk, n_z, len(ells))