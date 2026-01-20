# Limit number of cores
import os
os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Import packages
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import h5py
import getdist
import logging
import pandas as pd
from nautilus import Prior, Sampler
from getdist import plots, MCSamples
from pathlib import Path

rootpath = Path('/home/dneup16/leiden_phd/scripts/GNAT')
if str(rootpath) not in os.sys.path:
    os.sys.path.append(str(rootpath))

import src.galaxy_alignment_prediction_tool.multipoles
import src.galaxy_alignment_prediction_tool.powerspectrum
import src.galaxy_alignment_prediction_tool.projections
import src.galaxy_alignment_prediction_tool.fitter
import src.galaxy_alignment_prediction_tool.utils

# Reload to update changes
import importlib
importlib.reload(src.galaxy_alignment_prediction_tool.multipoles)
importlib.reload(src.galaxy_alignment_prediction_tool.powerspectrum)
importlib.reload(src.galaxy_alignment_prediction_tool.projections)
importlib.reload(src.galaxy_alignment_prediction_tool.fitter)
importlib.reload(src.galaxy_alignment_prediction_tool.utils)

from src.galaxy_alignment_prediction_tool.multipoles import multipoles
from src.galaxy_alignment_prediction_tool.powerspectrum import powerSpectrum
from src.galaxy_alignment_prediction_tool.projections import projections
from src.galaxy_alignment_prediction_tool.fitter import fitter
from src.galaxy_alignment_prediction_tool.utils import ioUtils

# TODO: Make config file read in

# Initialise classes
powerSpectrumHandler = powerSpectrum()
multipolesHandler = multipoles()
ioUtilsHandler = ioUtils()
projectionsHandler = projections()

def get_predictions(
    cosmology: ccl.Cosmology,
    redshift: float,
    k_input: np.ndarray,
    rp: np.ndarray,
    pimax: float,
):
    pkmm_ccl = powerSpectrumHandler.ccl_power_spectrum(
        cosmo=cosmology,
        k_input=k_input,
        redshift=redshift,
        pk_type='nonlinear_matter' # TODO: CAREFUL!!
    )

    # Initialise power spectra
    pkgp = powerSpectrumHandler.convert_pk_matter_to_pk_gplus(
        pk_matter=pkmm_ccl,
        cosmo=cosmology,
        redshift=redshift
    )
    pkgg = powerSpectrumHandler.convert_pk_matter_to_pk_gg(
        pk_matter=pkmm_ccl,
        cosmo=cosmology,
        redshift=redshift,
    )

    # Create splines
    pkgp_spline = powerSpectrumHandler.power_spectrum_spline(
        k_input=k_input,
        pk_input=pkgp,
    )
    pkgg_spline = powerSpectrumHandler.power_spectrum_spline(
        k_input=k_input,
        pk_input=pkgg,
    )

    # Calculate multipoles
    r_xigp_list, xigp_list, ells_gp = multipolesHandler.return_all_multipoles_from_power_spectrum(
        power_spectrum_spline=pkgp_spline,
        probe_type='g+',
    )

    r_xigg_list, xigg_list, ells_gg = multipolesHandler.return_all_multipoles_from_power_spectrum(
        power_spectrum_spline=pkgg_spline,
        probe_type='gg',
    )

    # Calculate projections
    w_gp = projectionsHandler.convert_multipoles_to_projected_correlation_function(
        r_list=r_xigp_list,
        xi_ell_list=xigp_list,
        ell_list=ells_gp,
        spin=2,
        rp_array=rp,
        pi_max=pimax,
    )

    w_gg = projectionsHandler.convert_multipoles_to_projected_correlation_function(
        r_list=r_xigg_list,
        xi_ell_list=xigg_list,
        ell_list=ells_gg,
        spin=0,
        rp_array=rp,
        pi_max=pimax,
    )

    return w_gp, w_gg, xigp_list[0], xigg_list[0], r_xigp_list[0], r_xigg_list[0]

def zeropad_covariance_matrix(top_left, bottom_right):
    zero_crosscovar = np.zeros((top_left.shape[0], bottom_right.shape[1]))
    full_covariance = np.block([
        [top_left, zero_crosscovar],
        [zero_crosscovar.T, bottom_right],
    ])
    return full_covariance

def produce_results_for_input_data(
    r_data_input,
    data_input,
    r_model_input,
    model_input,
    cov_input,
    fitting_range: list,
    projection_type_list=['gg', 'gp'],
    renormalise_input=True,
    chi2_from_svd=True,
    n_jk=125,
    outpath=None,
    outfile=None,
    logger=logging.getLogger(__name__),
):
    fitterHandler = fitter(logger=logger)

    def log_likelihood_all(params):
        A_IA = params['A_IA']
        b_g = params['b_g']

        chi2_total = fitterHandler.get_chi2_full_covariance(
            A_IA, 
            b_g, 
            r_data = r_data_input, 
            data = data_input, 
            r_model = r_model_input,
            model = model_input,
            cov = cov_input,
            fitting_range=fitting_range, 
            projection_type_list=projection_type_list,
            renormalise_input=renormalise_input,
            chi2_from_svd=chi2_from_svd,
            n_jk=n_jk,
        )

        return -0.5 * chi2_total

    prior = Prior()
    prior.add_parameter('A_IA', dist=(0, 20))
    prior.add_parameter('b_g', dist=(0, 10))

    sampler = Sampler(prior, log_likelihood_all, n_live=1000)
    sampler.run(verbose=True, discard_exploration=True)
    points, log_w, log_l = sampler.posterior()

    weights = np.exp(log_w - np.max(log_w))  # Normalize for numerical stability
    weights /= np.sum(weights)
    posterior_mean = np.average(points, axis=0, weights=weights)
    posterior_var = np.average((points - posterior_mean)**2, axis=0, weights=weights)
    posterior_std = np.sqrt(posterior_var)
    best_fit_paramsg = dict(zip(prior.keys, posterior_mean))

    logger.info(f'Posterior:')
    logger.info(f'A_IA = {best_fit_paramsg["A_IA"]} ± {posterior_std[0]}')
    logger.info(f'b_g = {best_fit_paramsg["b_g"]} ± {posterior_std[1]}')

    # Define names and labels programmatically (do this only once if they are same for all boxes)
    names = ['A_1', 'b_1']
    labels =['A_1', 'b_1']
    weights = np.exp(log_w - np.max(log_w))
    weights /= np.sum(weights)
    mcs = MCSamples(samples=points, names=names, labels=labels, weights=weights)
    if outpath is not None:
        outpath_mcmc = f'{outpath}/mcmc'
        mcs.saveAsText(outpath_mcmc)

        if outfile is not None: # optionally save triangle plot
            mcs = getdist.loadMCSamples(outpath_mcmc)
            g = plots.get_subplot_plotter()

            g.triangle_plot(
                mcs,
                labels,
                filled=False  # can be True, but might obscure overlaps
            )
            plt.savefig(outfile)
            plt.close()

    return best_fit_paramsg, posterior_std

def plot_best_fit_vs_data(
    best_fit_paramsg,
    redshift,
    r_redshift_list_data,
    measurement_redshift_list_data,
    measurement_cov_redshift_list_data,
    r_redshift_list_model,
    measurement_redshift_list_model,
    fitting_range,
    outpath,
    r_scaling_for_plot=1,
    which_measurement='projections',
):

    if which_measurement == 'projections':
        ax0_title = 'gg projection'
        ax1_title = 'gp projection'
        
        ax_xlabel = r'$r_\mathrm{p}$ [Mpc]'
        ax0_ylabel = r'$r_\mathrm{p} w_\mathrm{gg}$ [Mpc$^2$]'
        ax1_ylabel = r'$r_\mathrm{p} w_\mathrm{g+}$ [Mpc$^2$]'
    else:
        ax0_title = 'gg multipole'
        ax1_title = 'gp multipole'

        ax_xlabel = r'$r$ [Mpc]'
        ax0_ylabel = r'$r^2 \xi_\mathrm{gg}$ [Mpc$^2$]'
        ax1_ylabel = r'$r^2 \xi_\mathrm{g+}$ [Mpc$^2$]'

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel 1: gg ----
    best_fit_scaling_gg = best_fit_paramsg['b_g']**2
    ax[0].errorbar(
        r_redshift_list_data[0],
        r_redshift_list_data[0]**r_scaling_for_plot * measurement_redshift_list_data[0],
        yerr = np.sqrt(np.diag(measurement_cov_redshift_list_data[0])) * r_redshift_list_data[0]**r_scaling_for_plot,
        fmt='o',
    )
    ax[0].plot(
        r_redshift_list_model[0],
        r_redshift_list_model[0]**r_scaling_for_plot * measurement_redshift_list_model[0] * best_fit_scaling_gg,
        '-'
    )
    ax[0].set_title(ax0_title)
    ax[0].set_xlabel(ax_xlabel)
    ax[0].set_ylabel(ax0_ylabel)

    # ---- Panel 2: gg ----
    best_fit_scaling_gp = best_fit_paramsg['A_IA'] * best_fit_paramsg['b_g']
    ax[1].errorbar(
        r_redshift_list_data[1],
        r_redshift_list_data[1]**r_scaling_for_plot * measurement_redshift_list_data[1],
        yerr = np.sqrt(np.diag(measurement_cov_redshift_list_data[1])) * r_redshift_list_data[1]**r_scaling_for_plot,
        fmt='o',
    )
    ax[1].plot(
        r_redshift_list_model[1],
        r_redshift_list_model[1]**r_scaling_for_plot * measurement_redshift_list_model[1] * best_fit_scaling_gp,
        '-'
    )
    ax[1].set_title(ax1_title)
    ax[1].set_xlabel(ax_xlabel)
    ax[1].set_ylabel(ax1_ylabel)

    for idx, ax_ in enumerate(ax):
        ymin = 0.9*min(r_redshift_list_data[idx]**r_scaling_for_plot * measurement_redshift_list_data[idx])
        ymax = 1.1*max(r_redshift_list_data[idx]**r_scaling_for_plot * measurement_redshift_list_data[idx])
        ax_.fill_betweenx(
            y=[ymin, ymax],
            x1=[min(r_redshift_list_data[idx]),min(r_redshift_list_data[idx])],
            x2=[fitting_range[0], fitting_range[0]],
            color='gray',
            alpha=0.3,
            label='Fitting range'
        )
        ax_.set_xlim(min(r_redshift_list_data[idx])*0.95,max(r_redshift_list_data[idx])*1.05)
        ax_.set_ylim(ymin, ymax)    
        ax_.set_xscale('log')

    plt.suptitle(f'Best-fit model vs data for redshift {redshift}')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():

    # Initialise logger
    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Initialise cosmology
    cosmo_dict = {
        'L400_m7': {
            'h': 0.681,
            'Omega_c': 0.256011,
            'Omega_b': 0.0486,
            'm_nu': [0.06, 0.0, 0.0],
            'mass_split': 'list',
            'A_s': 2.099e-9,
            'n_s': 0.967,
        },
        'TNG300': {
            'h': 0.6774,
            'Omega_c': 0.3089-0.0486,
            'Omega_b': 0.0486,
            'sigma8': 0.8159,
            'n_s': 0.9667,
        }
    }
    k_input = np.geomspace(1e-5, 500, 1000)

    # Load in data
    # sample_str = 'D_nstar_gt50_S_mstar_gt9p27'
    sample_str = 'D_nstar_gt50_S_mstar_gt9p27_mDM_gt11p34'
    parent = Path('/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/')
    path_to_h5py = parent / 'run_20260120' / 'IA_data_modeling_z_evolution_DM.hdf5'
    outpath = parent / 'run_20260120_LA' / 'DM_gt9p27_mDM_gt11p34'
    output_csv = 'IA_fitting_results_summary_DM_gt9p27_mDM_gt11p34.csv'
    os.makedirs(str(outpath), exist_ok=True)

    measurement_dict = ioUtilsHandler.load_h5_recursive(
        file_path=str(path_to_h5py)
    )
    logger.info('Loaded in measurement data.')

    # Iterate over simulations and snapshots
    data_str = {
        'L400_m7': f'{sample_str}_LOSy_{sample_str}_LOSz',
        # 'TNG300': f'{sample_str}_LOSy_{sample_str}_LOSz',
        'TNG300': f'{sample_str}_LOSz'
    }

    redshifts = {
        'L400_m7': {'Snapshot_127': 0, 'Snapshot_102': 0.5, 'Snapshot_92': 1, 'Snapshot_84': 1.5, 'Snapshot_76': 2, 'Snapshot_68': 2.5},
        'TNG300': {'Snapshot_33': 2.0, 'Snapshot_39': 1.53, 'Snapshot_40': 1.5, 'Snapshot_50': 1., 'Snapshot_67': 0.5, 'Snapshot_99': 0.}
    }

    logger.info('The following simulations and snapshots are available in the h5py:')
    for sim in measurement_dict.keys():
        logger.info(f'Simulation: {sim}')
        for snapshot in measurement_dict[sim].keys():
            logger.info(f' - Snapshot: {snapshot}')

    columns = ['simulation', 'snapshot', 'redshift', 'A_IA', 'A_IA_err', 'b_g', 'b_g_err', 'estimator']
    output_df = pd.DataFrame(columns=columns)
    for sim in measurement_dict.keys():
        # if sim == 'L400_m7':
        #     continue # Skip L400_m7 for testing

        # Specify projection parameters
        rmin = 0.1/cosmo_dict[sim]['h']  # [Mpc]
        rmax = 50./cosmo_dict[sim]['h']  # [Mpc]
        num_r = 50
        pi_max = 50./cosmo_dict[sim]['h']  # [Mpc]
        rp = np.geomspace(rmin, rmax, num_r)

        for snapshot in measurement_dict[sim].keys():

            redshift_sim = redshifts[sim]
            
            will_analyse = ioUtilsHandler.is_in_snapshot_dict(
                snapshot_dict=redshift_sim,
                key=snapshot,
            )

            if will_analyse:
                logger.info(f'Analysing simulation {sim}, snapshot {snapshot} at redshift {redshift_sim[snapshot]}')
                # print(measurement_dict[sim][snapshot]['w_gg'].keys())
                try:
                    rp_gg_data = measurement_dict[sim][snapshot]['w_gg'][data_str[sim] + '_rp']/cosmo_dict[sim]['h']
                    w_gg_data = measurement_dict[sim][snapshot]['w_gg'][data_str[sim]]/cosmo_dict[sim]['h']
                    w_gg_cov_data = measurement_dict[sim][snapshot]['w_gg'][data_str[sim] + '_cov']/cosmo_dict[sim]['h']**2

                    rp_gplus_data = measurement_dict[sim][snapshot]['w_g_plus'][data_str[sim] + '_rp']/cosmo_dict[sim]['h']
                    w_gplus_data = measurement_dict[sim][snapshot]['w_g_plus'][data_str[sim]]/cosmo_dict[sim]['h']
                    w_gplus_cov_data = measurement_dict[sim][snapshot]['w_g_plus'][data_str[sim] + '_cov']/cosmo_dict[sim]['h']**2

                    r_gg_data = measurement_dict[sim][snapshot]['multipoles_gg'][data_str[sim] + '_r']/cosmo_dict[sim]['h']
                    xi_gg_data = measurement_dict[sim][snapshot]['multipoles_gg'][data_str[sim]]
                    xi_gg_cov_data = measurement_dict[sim][snapshot]['multipoles_gg'][data_str[sim] + '_cov']

                    r_gplus_data = measurement_dict[sim][snapshot]['multipoles_g_plus'][data_str[sim] + '_r']/cosmo_dict[sim]['h']
                    xi_gplus_data = measurement_dict[sim][snapshot]['multipoles_g_plus'][data_str[sim]]
                    xi_gplus_cov_data = measurement_dict[sim][snapshot]['multipoles_g_plus'][data_str[sim] + '_cov']

                except KeyError as e:
                    logger.warning(f'Could not extract all data for simulation {sim}, snapshot {snapshot}: {e}')
                    logger.warning('Skipping to next snapshot.')
                    continue

                # Initialise cosmology and predictions
                cosmo = ccl.Cosmology(**cosmo_dict[sim])
                w_gplus_model, w_gg_model, xi_gplus_model, xi_gg_model, r_gplus_model, r_gg_model = get_predictions(
                    cosmology=cosmo,
                    redshift=redshift_sim[snapshot],
                    k_input=k_input,
                    rp=rp,
                    pimax=pi_max,
                )

                # Fill up off diagonals with zeros
                projections_cov = zeropad_covariance_matrix(w_gg_cov_data, w_gplus_cov_data)
                multipoles_cov = zeropad_covariance_matrix(xi_gg_cov_data, xi_gplus_cov_data)

                # Produce results for projections
                outpath_projections = outpath / f'fit_results_projections_{sim}' / f'fit_results_{snapshot}'
                outpath_multipoles = outpath / f'fit_results_multipoles_{sim}' / f'fit_results_{snapshot}'
                outpath_projection_plots = outpath / f'fit_results_projections_{sim}' / 'plots'
                outpath_multipole_plots = outpath / f'fit_results_multipoles_{sim}' / 'plots'

                # Make folders
                os.makedirs(str(outpath_projections), exist_ok=True)
                os.makedirs(str(outpath_multipoles), exist_ok=True)
                os.makedirs(str(outpath_projection_plots), exist_ok=True)
                os.makedirs(str(outpath_multipole_plots), exist_ok=True)

                # Fit data
                fitting_range = [6./cosmo_dict[sim]['h'], 50./cosmo_dict[sim]['h']]

                try:
                    logger.info('Starting MCMC for projections...')
                    best_fit_params_projections, posterior_std_projections = produce_results_for_input_data(
                        r_data_input=[rp_gg_data, rp_gplus_data],
                        data_input=[w_gg_data, w_gplus_data],
                        r_model_input=[rp, rp],
                        model_input=[w_gg_model, w_gplus_model],
                        cov_input=projections_cov,
                        fitting_range=fitting_range,
                        projection_type_list=['gg', 'gp'],
                        renormalise_input=True,
                        chi2_from_svd=True,
                        n_jk=125,
                        outpath=outpath_projections,
                        outfile=str(outpath_projection_plots / f'mcmc_triangle_proj_{snapshot}.png')
                    )

                    plot_best_fit_vs_data(
                        best_fit_paramsg=best_fit_params_projections,
                        r_redshift_list_data=[rp_gg_data, rp_gplus_data],
                        measurement_redshift_list_data=[w_gg_data, w_gplus_data],
                        measurement_cov_redshift_list_data=[w_gg_cov_data, w_gplus_cov_data],
                        r_redshift_list_model=[rp, rp],
                        measurement_redshift_list_model=[w_gg_model, w_gplus_model],
                        fitting_range=fitting_range,
                        outpath=str(outpath_projection_plots / f'best_fit_vs_data_proj_{snapshot}.png'),
                        r_scaling_for_plot=1,
                        which_measurement='projections',
                        redshift=redshift_sim[snapshot],
                    )
                    logger.info('...done')

                except Exception as e:
                    logger.error(f'Error during projection fitting for simulation {sim}, snapshot {snapshot}: {e}')
                    logger.error('Skipping to next multipoles and saving NaNs.')
                    
                    best_fit_params_projections = {'A_IA': np.nan, 'b_g': np.nan}
                    posterior_std_projections = [np.nan, np.nan]

                try:
                    logger.info('Starting MCMC for multipoles...')
                    best_fit_params_multipoles, posterior_std_multipoles = produce_results_for_input_data(
                        r_data_input=[r_gg_data, r_gplus_data],
                        data_input=[xi_gg_data, xi_gplus_data],
                        r_model_input=[r_gg_model, r_gplus_model],
                        model_input=[xi_gg_model, xi_gplus_model],
                        cov_input=multipoles_cov,
                        fitting_range=fitting_range,
                        projection_type_list=['gg', 'gp'],
                        renormalise_input=True,
                        chi2_from_svd=True,
                        n_jk=125,
                        outpath=outpath_multipoles,
                        outfile=str(outpath_multipole_plots / f'mcmc_triangle_multpl_{snapshot}.png')
                    )

                    plot_best_fit_vs_data(
                        best_fit_paramsg=best_fit_params_multipoles,
                        r_redshift_list_data=[r_gg_data, r_gplus_data],
                        measurement_redshift_list_data=[xi_gg_data, xi_gplus_data],
                        measurement_cov_redshift_list_data=[xi_gg_cov_data, xi_gplus_cov_data],
                        r_redshift_list_model=[r_gg_model, r_gplus_model],
                        measurement_redshift_list_model=[xi_gg_model, xi_gplus_model],
                        fitting_range=fitting_range,
                        outpath=str(outpath_multipole_plots / f'best_fit_vs_data_multpl_{snapshot}.png'),
                        r_scaling_for_plot=2,
                        which_measurement='multipoles',
                        redshift=redshift_sim[snapshot],
                    )
                    logger.info('...done')

                except Exception as e:
                    logger.error(f'Error during multiple fitting for simulation {sim}, snapshot {snapshot}: {e}')
                    logger.error('Skipping to next snapshot and saving NaNs.')
                    
                    best_fit_params_multipoles = {'A_IA': np.nan, 'b_g': np.nan}
                    posterior_std_multipoles = [np.nan, np.nan]

                # Save to output dataframe
                output_df = pd.concat([output_df, pd.DataFrame({
                    'simulation': [sim, sim],
                    'snapshot': [snapshot, snapshot],
                    'redshift': [redshift_sim[snapshot], redshift_sim[snapshot]],
                    'A_IA': [best_fit_params_projections['A_IA'], best_fit_params_multipoles['A_IA']],
                    'A_IA_err': [posterior_std_projections[0], posterior_std_multipoles[0]],
                    'b_g': [best_fit_params_projections['b_g'], best_fit_params_multipoles['b_g']],
                    'b_g_err': [posterior_std_projections[1], posterior_std_multipoles[1]],
                    'estimator': ['projections', 'multipoles'],
                })], ignore_index=True)

    # Save output dataframe to csv
    output_df.to_csv(
        path_or_buf=outpath.parent / output_csv,
        sep='\t',
        index=False,
    )

if __name__ == '__main__':
    main()