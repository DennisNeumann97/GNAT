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
    prior.add_parameter('A_IA', dist=(0, 50))
    prior.add_parameter('b_g', dist=(0, 15))

    sampler = Sampler(prior, log_likelihood_all, n_live=1000)
    sampler.run(verbose=True, discard_exploration=True)
    points, log_w, log_l = sampler.posterior()

    # Get posterior mean and std (weighted average)
    weights = np.exp(log_w - np.max(log_w))  # Normalize for numerical stability
    weights /= np.sum(weights)
    posterior_mean = np.average(points, axis=0, weights=weights)
    posterior_var = np.average((points - posterior_mean)**2, axis=0, weights=weights)
    posterior_std = np.sqrt(posterior_var)
    best_fit_paramsg = dict(zip(prior.keys, posterior_mean))

    # Get final chi^2 at best-fit parameters
    chi2_best_fit = -2*log_likelihood_all(best_fit_paramsg)
    n_fitpoints = fitterHandler.return_npoints_after_SVD(
        n_jk=n_jk,
        cov=cov_input,
    )
    dof = n_fitpoints - len(prior.keys)
    reduced_chi2 = chi2_best_fit / dof

    logger.info(f'Posterior:')
    logger.info(f'A_IA = {best_fit_paramsg["A_IA"]} ± {posterior_std[0]}')
    logger.info(f'b_g = {best_fit_paramsg["b_g"]} ± {posterior_std[1]}')
    logger.info(f'Total chi^2 = {chi2_best_fit}, dof = {dof}, reduced chi^2 = {reduced_chi2}')

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

    return best_fit_paramsg, posterior_std, reduced_chi2

def plot_best_fit_vs_data(
    best_fit_paramsg,
    redshift,
    r_redshift_list_data,
    measurement_redshift_list_data,
    measurement_cov_redshift_data,
    r_redshift_list_model,
    measurement_redshift_list_model,
    fitting_range,
    outpath,
    red_chi2,
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

    # Cut covariance matrix for error bar display
    cov_gg = np.diag(measurement_cov_redshift_data)[0:len(r_redshift_list_data[0])]
    cov_gp = np.diag(measurement_cov_redshift_data)[len(r_redshift_list_data[0]):]

    # ---- Panel 1: gg ----
    best_fit_scaling_gg = best_fit_paramsg['b_g']**2
    ax[0].errorbar(
        r_redshift_list_data[0],
        r_redshift_list_data[0]**r_scaling_for_plot * measurement_redshift_list_data[0],
        yerr = np.sqrt(cov_gg) * r_redshift_list_data[0]**r_scaling_for_plot,
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
        yerr = np.sqrt(cov_gp) * r_redshift_list_data[1]**r_scaling_for_plot,
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

    plt.suptitle(f'Best-fit model vs data for redshift {redshift}, reduced chi^2 = {red_chi2:.2f}')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def setup_logger():
    """Initialize and configure the logger."""
    logger = logging.getLogger(__name__)
    logger.handlers.clear()

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

def get_cosmology_configs():
    """Return cosmology configurations for different simulations."""
    return {
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


def get_redshift_maps():
    """Return redshift mappings for each simulation.
    
    Returns:
        dict: Snapshot to redshift mapping for each simulation
    """
    return {
        'L400_m7': {
            'Snapshot_127': 0, 'Snapshot_102': 0.5, 'Snapshot_92': 1,
            'Snapshot_84': 1.5, 'Snapshot_76': 2, 'Snapshot_68': 2.5
        },
        'TNG300': {
            'Snapshot_33': 2.0, 'Snapshot_39': 1.53, 'Snapshot_40': 1.5,
            'Snapshot_50': 1., 'Snapshot_67': 0.5, 'Snapshot_99': 0.
        }
    }

def get_data_configs(
    pos_sample='nstar_gt50',
    shape_sample='mstar_gt9p27_mDM_gt11p34',
    probe='DM',
    input_path=None,
    output_path=None,
):
    """Return data configuration parameters.
    
    Args:
        pos_sample: Sample definition for position/clustering
        shape_sample: Sample definition for shape/alignment
        probe: Probe type (e.g., 'DM')
        los_suffix: Line-of-sight suffix list (Options: ['_LOSy', '_LOSz'], ['_LOSz'])
        input_path: Input directory path. If None, uses default.
        output_path: Output directory path. If None, uses default.
    
    Returns:
        dict: Configuration dictionary with paths, data strings, and redshift mappings
    """
    # Set default paths if not provided
    if input_path is None:
        input_path = Path('/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/run_20260303/')
    else:
        input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path('/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/run_20260303/')
    else:
        output_path = Path(output_path)
    
    
    # Get redshift mappings
    redshifts = get_redshift_maps()
    
    # File paths - incorporate sample definitions into output directory name
    path_to_h5py = input_path / f'IA_data_measurement_z_evolution_{probe}.hdf5'
    
    # Create descriptive output directory name with sample info
    sample_tag = f'{pos_sample}_{shape_sample}'
    outpath = output_path / f'{probe}_{sample_tag}'
    output_csv = f'IA_fitting_results_summary_{probe}_{sample_tag}.csv'
    
    return {
        'path_to_h5py': path_to_h5py,
        'outpath': outpath,
        'output_csv': output_csv,
        'redshifts': redshifts,
    }

def load_measurement_data(path_to_h5py, logger):
    """Load measurement data from HDF5 file."""
    measurement_dict = ioUtilsHandler.load_h5_recursive(file_path=str(path_to_h5py))
    logger.info('Loaded in measurement data.')
    
    logger.info('The following simulations and snapshots are available in the h5py:')
    for sim in measurement_dict.keys():
        logger.info(f'Simulation: {sim}')
        for snapshot in measurement_dict[sim].keys():
            logger.info(f' - Snapshot: {snapshot}')
    
    return measurement_dict

def correct_color_cut_in_data_str(
    p_string, 
    colibre_color_cuts, 
    snapshot,
    n_projection=1,
):
    """Correct the color cut in the data string based on the snapshot."""

    if '_ri_' in p_string:
        p_string_split = p_string.split('_ri_')

        # Correct color cut value for all projections
        for idx_proj in range(n_projection):
            p_string_split[2*idx_proj+1] = p_string_split[2*idx_proj+1][:2] + str(colibre_color_cuts[snapshot])
            
        p_string = '_ri_'.join(p_string_split)
    
    return p_string

def extract_snapshot_data(
    measurement_dict,
    cosmo_dict,
    colibre_color_cuts,
    sim,
    snapshot,
    g_string,
    p_string,
    n_projection: int=2,
):
    
    # Define data strings
    # ------------------------------------------------
    # Exchange colour cut with redshift dependent value if present in p_string
    p_string = correct_color_cut_in_data_str(
        p_string=p_string,
        colibre_color_cuts=colibre_color_cuts,
        snapshot=snapshot,
        n_projection=n_projection
    )

    if n_projection == 2:
        gg_string = f'D_{g_string}_S_{g_string}_LOSy_D_{g_string}_S_{g_string}_LOSz'
        gplus_string = f'D_{g_string}_S_{p_string}_LOSy_D_{g_string}_S_{p_string}_LOSz'
    elif n_projection == 1:
        gg_string = f'D_{g_string}_S_{g_string}_LOSz'
        gplus_string = f'D_{g_string}_S_{p_string}_LOSz'
    else:
        raise ValueError(f'Invalid n_projection value: {n_projection}. Must be 1 or 2.')

    cov_string = f'D_{g_string}_S_{p_string}_cov'
    # ------------------------------------------------ 

    # Load data
    # ------------------------------------------------
    output_dict = {
        # projections
        'rp_gg_data': measurement_dict[sim][snapshot]['w_gg'][gg_string + '_rp']/cosmo_dict[sim]['h'],
        'w_gg_data': measurement_dict[sim][snapshot]['w_gg'][gg_string]/cosmo_dict[sim]['h'],
        'rp_gplus_data': measurement_dict[sim][snapshot]['w_g_plus'][gplus_string + '_rp']/cosmo_dict[sim]['h'],
        'w_gplus_data': measurement_dict[sim][snapshot]['w_g_plus'][gplus_string]/cosmo_dict[sim]['h'],

        # multipoles
        'r_gg_data': measurement_dict[sim][snapshot]['multipoles_gg'][gg_string + '_r']/cosmo_dict[sim]['h'],
        'xi_gg_data': measurement_dict[sim][snapshot]['multipoles_gg'][gg_string]/cosmo_dict[sim]['h'],
        'r_gplus_data': measurement_dict[sim][snapshot]['multipoles_g_plus'][gplus_string + '_r']/cosmo_dict[sim]['h'],
        'xi_gplus_data': measurement_dict[sim][snapshot]['multipoles_g_plus'][gplus_string]/cosmo_dict[sim]['h'],

        # covariance matrix
        'w_cov_data': measurement_dict[sim][snapshot]['w'][cov_string]/cosmo_dict[sim]['h']**2,
        'xi_cov_data': measurement_dict[sim][snapshot]['multipoles'][cov_string]/cosmo_dict[sim]['h']**2
    }
    # ------------------------------------------------

    return output_dict

def analyze_snapshot(
    sim, snapshot, redshift, data, cosmo_dict, k_input, rp, pi_max, 
    fitting_range, outpath, logger
):
    """Analyze a single snapshot: fit data and create plots.
    
    Returns:
        tuple: (best_fit_params_projections, posterior_std_projections,
                best_fit_params_multipoles, posterior_std_multipoles)
    """
    # Initialize cosmology and get predictions
    cosmo = ccl.Cosmology(**cosmo_dict[sim])
    w_gplus_model, w_gg_model, xi_gplus_model, xi_gg_model, r_gplus_model, r_gg_model = get_predictions(
        cosmology=cosmo,
        redshift=redshift,
        k_input=k_input,
        rp=rp,
        pimax=pi_max,
    )

    # Setup output paths
    outpath_projections = outpath / f'fit_results_projections_{sim}' / f'fit_results_{snapshot}'
    outpath_multipoles = outpath / f'fit_results_multipoles_{sim}' / f'fit_results_{snapshot}'
    outpath_projection_plots = outpath / f'fit_results_projections_{sim}' / 'plots'
    outpath_multipole_plots = outpath / f'fit_results_multipoles_{sim}' / 'plots'
    
    os.makedirs(str(outpath_projections), exist_ok=True)
    os.makedirs(str(outpath_multipoles), exist_ok=True)
    os.makedirs(str(outpath_projection_plots), exist_ok=True)
    os.makedirs(str(outpath_multipole_plots), exist_ok=True)
    
    # Fit projections
    try:
        logger.info('Starting MCMC for projections...')
        best_fit_params_projections, posterior_std_projections, reduced_chi2_projections = produce_results_for_input_data(
            r_data_input=[data['rp_gg_data'], data['rp_gplus_data']],
            data_input=[data['w_gg_data'], data['w_gplus_data']],
            r_model_input=[rp, rp],
            model_input=[w_gg_model, w_gplus_model],
            cov_input=data['w_cov_data'],
            fitting_range=fitting_range,
            projection_type_list=['gg', 'gp'],
            renormalise_input=True,
            chi2_from_svd=True,
            n_jk=125,
            outpath=outpath_projections,
            outfile=str(outpath_projection_plots / f'mcmc_triangle_proj_{snapshot}.png'),
            logger=logger,
        )
        
        plot_best_fit_vs_data(
            best_fit_paramsg=best_fit_params_projections,
            r_redshift_list_data=[data['rp_gg_data'], data['rp_gplus_data']],
            measurement_redshift_list_data=[data['w_gg_data'], data['w_gplus_data']],
            measurement_cov_redshift_data=data['w_cov_data'],
            r_redshift_list_model=[rp, rp],
            measurement_redshift_list_model=[w_gg_model, w_gplus_model],
            fitting_range=fitting_range,
            outpath=str(outpath_projection_plots / f'best_fit_vs_data_proj_{snapshot}.png'),
            r_scaling_for_plot=1,
            which_measurement='projections',
            redshift=redshift,
            red_chi2=reduced_chi2_projections,
        )
        logger.info('...done')
    except Exception as e:
        logger.error(f'Error during projection fitting: {e}')
        logger.error('Saving NaNs for projections.')
        best_fit_params_projections = {'A_IA': np.nan, 'b_g': np.nan}
        posterior_std_projections = [np.nan, np.nan]
    
    # Fit multipoles
    try:
        logger.info('Starting MCMC for multipoles...')
        best_fit_params_multipoles, posterior_std_multipoles, reduced_chi2_multipoles = produce_results_for_input_data(
            r_data_input=[data['r_gg_data'], data['r_gplus_data']],
            data_input=[data['xi_gg_data'], data['xi_gplus_data']],
            r_model_input=[r_gg_model, r_gplus_model],
            model_input=[xi_gg_model, xi_gplus_model],
            cov_input=data['xi_cov_data'],
            fitting_range=fitting_range,
            projection_type_list=['gg', 'gp'],
            renormalise_input=True,
            chi2_from_svd=True,
            n_jk=125,
            outpath=outpath_multipoles,
            outfile=str(outpath_multipole_plots / f'mcmc_triangle_multpl_{snapshot}.png'),
            logger=logger,
        )
        
        plot_best_fit_vs_data(
            best_fit_paramsg=best_fit_params_multipoles,
            r_redshift_list_data=[data['r_gg_data'], data['r_gplus_data']],
            measurement_redshift_list_data=[data['xi_gg_data'], data['xi_gplus_data']],
            measurement_cov_redshift_data=data['xi_cov_data'],
            r_redshift_list_model=[r_gg_model, r_gplus_model],
            measurement_redshift_list_model=[xi_gg_model, xi_gplus_model],
            fitting_range=fitting_range,
            outpath=str(outpath_multipole_plots / f'best_fit_vs_data_multpl_{snapshot}.png'),
            r_scaling_for_plot=2,
            which_measurement='multipoles',
            redshift=redshift,
            red_chi2=reduced_chi2_multipoles,
        )
        logger.info('...done')
    except Exception as e:
        logger.error(f'Error during multipole fitting: {e}')
        logger.error('Saving NaNs for multipoles.')
        best_fit_params_multipoles = {'A_IA': np.nan, 'b_g': np.nan}
        posterior_std_multipoles = [np.nan, np.nan]
    
    return (
        best_fit_params_projections, posterior_std_projections, reduced_chi2_projections,
        best_fit_params_multipoles, posterior_std_multipoles, reduced_chi2_multipoles,
    )

def iterate_over_simulations_and_snapshots(
    g_string: str,
    p_string: str,
    sims: list,
    n_projection: int,
    measurement_dict: dict,
    cosmo_dict: dict,
    colibre_color_cuts: dict,
    k_input: np.ndarray,
    pi_max: dict, 
    data_config: dict, 
    logger: logging.Logger,
    fitting_range: list=None, 
    rp: np.ndarray=None,
):
    """Iterate over simulations and snapshots, analyze each snapshot, and collect results."""
    # Initialize results storage
    output_df_list = []

    for sim in sims:

        h = cosmo_dict[sim]['h']
        rmin = 0.1 / h  # [Mpc]
        rmax = 50. / h  # [Mpc]
        num_r = 50

        rp = np.geomspace(rmin, rmax, num_r) if rp is None else rp
        fitting_range = [6. / h, 50. / h] if fitting_range is None else fitting_range

        for snapshot in measurement_dict[sim].keys():
            redshift_sim = data_config['redshifts'][sim]
            will_analyse = ioUtilsHandler.is_in_snapshot_dict(snapshot_dict=redshift_sim, key=snapshot)
            
            if not will_analyse:
                continue
            
            redshift = redshift_sim[snapshot]
            logger.info(f'Analysing simulation {sim}, snapshot {snapshot} at redshift {redshift}')
            
            try:
                data = extract_snapshot_data(
                    measurement_dict=measurement_dict,
                    cosmo_dict=cosmo_dict,
                    colibre_color_cuts=colibre_color_cuts,
                    sim=sim, 
                    snapshot=snapshot,
                    g_string=g_string,
                    p_string=p_string,
                    n_projection=n_projection,
                )
            except KeyError as e:
                logger.warning(f'KeyError occurred: {e}')
                logger.warning('Skipping to next snapshot.')
                continue
            
            (
                best_fit_params_projections, posterior_std_projections, reduced_chi2_projections,
                best_fit_params_multipoles, posterior_std_multipoles, reduced_chi2_multipoles
            ) = analyze_snapshot(
                sim, snapshot, redshift, data, cosmo_dict, k_input, rp,
                pi_max[sim] / h, fitting_range, data_config['outpath'], logger
            )
            
            output_df_list.append(
                pd.DataFrame({
                    'simulation': [sim, sim],
                    'snapshot': [snapshot, snapshot],
                    'redshift': [redshift, redshift],
                    'A_IA': [best_fit_params_projections['A_IA'], best_fit_params_multipoles['A_IA']],
                    'A_IA_err': [posterior_std_projections[0], posterior_std_multipoles[0]],
                    'b_g': [best_fit_params_projections['b_g'], best_fit_params_multipoles['b_g']],
                    'b_g_err': [posterior_std_projections[1], posterior_std_multipoles[1]],
                    'reduced_chi2': [reduced_chi2_projections, reduced_chi2_multipoles],
                    'estimator': ['projections', 'multipoles'],
            }))

    # Save output dataframe to CSV
    output_df = pd.concat(output_df_list, ignore_index=True)
    output_df.to_csv(
        path_or_buf=data_config['outpath'].parent / data_config['output_csv'],
        sep='\t',
        index=False,
    )
    logger.info(f"Results saved to {data_config['outpath'].parent / data_config['output_csv']}")

def main():
    """Main analysis pipeline for IA redshift dependency."""
    # Setup
    logger = setup_logger()
    cosmo_dict = get_cosmology_configs()
    k_input = np.geomspace(1e-5, 500, 1000)
    input_path = '/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/run_20260311/'
    output_path = '/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/run_20260311_tests/'

    # Define measurement list
    sample_list = [
        ["mstar_gt9p27_mDM_gt11p34_ri_gt", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_ri_lt", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_q0", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mlt11", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mgt11", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mlt11", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mgt11", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mlt10", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mgt10", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mlt10", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mgt10", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0", "nstar_gt50"],
        ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0", "nstar_gt50"],
    ]
    probe_list = ['DM', 'stars']
    sim_list = ['L400_m7', 'TNG300']
    n_projection_list = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    # Projection parameters
    pi_max = {
        'L400_m7': 50.,
        'TNG300': 40.,
    }

    # Colibre color cuts for each snapshot (if needed for data string correction)
    colibre_color_cuts = {
        'Snapshot_127': 0.27099231, 'Snapshot_102': 0.23761419, 'Snapshot_92': 0.20026581, 'Snapshot_84': 0.1694618, 'Snapshot_76': 0.14663814, 'Snapshot_68': 0.09957861,
    }

    # Iterate over all measurements
    for probe in probe_list:
        for sample_idx in range(len(sample_list)):
            shape_sample, pos_sample = sample_list[sample_idx]
            n_projection = n_projection_list[sample_idx]

            logger.info(f'Processing probe {probe} with position sample "{pos_sample}" and shape sample "{shape_sample}" using {n_projection} projection(s).')

            data_config = get_data_configs(
                pos_sample=pos_sample,
                shape_sample=shape_sample,
                probe=probe,
                input_path=input_path,
                output_path=output_path,
            )

            # Create output directory
            os.makedirs(str(data_config['outpath']), exist_ok=True)
            
            # Load data
            measurement_dict = load_measurement_data(data_config['path_to_h5py'], logger)

            # Iterate over simulations and snapshots
            iterate_over_simulations_and_snapshots(
                sims=sim_list,
                g_string=pos_sample,
                p_string=shape_sample,
                n_projection=n_projection,
                measurement_dict=measurement_dict,
                cosmo_dict=cosmo_dict,
                colibre_color_cuts=colibre_color_cuts,
                k_input=k_input,
                pi_max=pi_max, 
                data_config=data_config, 
                logger=logger,
            )

if __name__ == '__main__':
    main()