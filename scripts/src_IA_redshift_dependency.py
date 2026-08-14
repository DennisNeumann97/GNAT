# Import packages
import os
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

# rootpath = Path('/home/dneup16/leiden_phd/scripts/GNAT')
# if str(rootpath) not in os.sys.path:
#     os.sys.path.append(str(rootpath))

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
		pk_type='nonlinear_matter'  # TODO: CAREFUL!!
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
			r_data=r_data_input,
			data=data_input,
			r_model=r_model_input,
			model=model_input,
			cov=cov_input,
			fitting_range=fitting_range,
			projection_type_list=projection_type_list,
			renormalise_input=renormalise_input,
			chi2_from_svd=chi2_from_svd,
			n_jk=n_jk,
		)

		return -0.5 * chi2_total

	# Initialise prior
	prior = Prior()
	prior.add_parameter('A_IA', dist=(0, 50))
	prior.add_parameter('b_g', dist=(0, 15))

	# Get dof first
	# ------------------------------------------------
	n_fitpoints = fitterHandler.return_n_fitpoints(
		r_data=r_data_input,
		cov=cov_input,
		fitting_range=fitting_range,
		renormalise_input=renormalise_input,
		chi2_from_svd=chi2_from_svd,
		n_jk=n_jk,
	)
	n_params = len(prior.keys)
	dof = n_fitpoints - n_params
	logger.info(f'Fitting {n_fitpoints} points with {n_params} parameters.')

	if dof <= 0:
		logger.error(f'N_dof <= 0... Skipping')
		best_fit_paramsg = np.ones(n_params) * np.nan
		posterior_std = np.ones(n_params) * np.nan
		reduced_chi2 = np.nan

		return best_fit_paramsg, posterior_std, reduced_chi2
	# ------------------------------------------------

	# Run sampler
	sampler = Sampler(prior, log_likelihood_all, n_live=1000, seed=20180403)
	sampler.run(verbose=True, discard_exploration=True)
	points, log_w, log_l = sampler.posterior()

	# Get posterior mean and std (weighted average)
	weights = np.exp(log_w - np.max(log_w))  # Normalize for numerical stability
	weights /= np.sum(weights)
	posterior_mean = np.average(points, axis=0, weights=weights)
	posterior_var = np.average((points - posterior_mean) ** 2, axis=0, weights=weights)
	posterior_std = np.sqrt(posterior_var)
	best_fit_paramsg = dict(zip(prior.keys, posterior_mean))

	# Get final chi^2 at best-fit parameters
	chi2_best_fit = - 2 * log_likelihood_all(best_fit_paramsg)
	reduced_chi2 = chi2_best_fit / dof

	logger.info(f'Posterior:')
	logger.info(f'A_IA = {best_fit_paramsg["A_IA"]} ± {posterior_std[0]}')
	logger.info(f'b_g = {best_fit_paramsg["b_g"]} ± {posterior_std[1]}')
	logger.info(f'Total chi^2 = {chi2_best_fit}, dof = {dof}, reduced chi^2 = {reduced_chi2}')

	# Define names and labels programmatically (do this only once if they are same for all boxes)
	names = ['A_1', 'b_1']
	labels = ['A_1', 'b_1']
	weights = np.exp(log_w - np.max(log_w))
	weights /= np.sum(weights)
	mcs = MCSamples(samples=points, names=names, labels=labels, weights=weights)
	if outpath is not None:
		outpath_mcmc = f'{outpath}/mcmc'
		mcs.saveAsText(outpath_mcmc)

		if outfile is not None:  # optionally save triangle plot
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
	best_fit_scaling_gg = best_fit_paramsg['b_g'] ** 2
	ax[0].errorbar(
		r_redshift_list_data[0],
		r_redshift_list_data[0] ** r_scaling_for_plot * measurement_redshift_list_data[0],
		yerr=np.sqrt(cov_gg) * r_redshift_list_data[0] ** r_scaling_for_plot,
		fmt='o',
	)
	ax[0].plot(
		r_redshift_list_model[0],
		r_redshift_list_model[0] ** r_scaling_for_plot * measurement_redshift_list_model[0] * best_fit_scaling_gg,
		'-'
	)
	ax[0].set_title(ax0_title)
	ax[0].set_xlabel(ax_xlabel)
	ax[0].set_ylabel(ax0_ylabel)

	# ---- Panel 2: gg ----
	best_fit_scaling_gp = best_fit_paramsg['A_IA'] * best_fit_paramsg['b_g']
	ax[1].errorbar(
		r_redshift_list_data[1],
		r_redshift_list_data[1] ** r_scaling_for_plot * measurement_redshift_list_data[1],
		yerr=np.sqrt(cov_gp) * r_redshift_list_data[1] ** r_scaling_for_plot,
		fmt='o',
	)
	ax[1].plot(
		r_redshift_list_model[1],
		r_redshift_list_model[1] ** r_scaling_for_plot * measurement_redshift_list_model[1] * best_fit_scaling_gp,
		'-'
	)
	ax[1].set_title(ax1_title)
	ax[1].set_xlabel(ax_xlabel)
	ax[1].set_ylabel(ax1_ylabel)

	for idx, ax_ in enumerate(ax):
		ymin = 0.9 * min(r_redshift_list_data[idx] ** r_scaling_for_plot * measurement_redshift_list_data[idx])
		ymax = 1.1 * max(r_redshift_list_data[idx] ** r_scaling_for_plot * measurement_redshift_list_data[idx])
		ax_.fill_betweenx(
			y=[ymin, ymax],
			x1=[min(r_redshift_list_data[idx]), min(r_redshift_list_data[idx])],
			x2=[fitting_range[0], fitting_range[0]],
			color='gray',
			alpha=0.3,
			label='Fitting range'
		)
		ax_.set_xlim(min(r_redshift_list_data[idx]) * 0.95, max(r_redshift_list_data[idx]) * 1.05)
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


#: Location of the simulation tables shared with the IA_z_evolution pipeline. That repo
#: generates the file from pipeline/registry.py, so cosmology, snapshot redshifts and
#: pi_max are defined in exactly one place. Override with the IA_SIM_TABLES environment
#: variable; if the file cannot be read, the built-in fallbacks below are used and a
#: warning is logged, so GNAT still runs standalone.
SIM_TABLES_PATH = os.environ.get(
	'IA_SIM_TABLES',
	'/Users/6918522/Documents/Work/IA_z_evolution/pipeline/sim_tables.yaml',
)

_FALLBACK_COSMOLOGY = {
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
		'Omega_c': 0.3089 - 0.0486,
		'Omega_b': 0.0486,
		'sigma8': 0.8159,
		'n_s': 0.9667,
	},
}

_FALLBACK_REDSHIFTS = {
	'L400_m7': {
		'Snapshot_127': 0, 'Snapshot_110': 0.3, 'Snapshot_102': 0.5, 'Snapshot_92': 1,
		'Snapshot_84': 1.5, 'Snapshot_76': 2, 'Snapshot_68': 2.5
	},
	'TNG300': {
		'Snapshot_33': 2.0, 'Snapshot_39': 1.53, 'Snapshot_40': 1.5,
		'Snapshot_50': 1., 'Snapshot_67': 0.5, 'Snapshot_78': 0.3, 'Snapshot_99': 0.
	},
}

_FALLBACK_PI_MAX = {'L400_m7': 50., 'TNG300': 40.}


def _load_sim_tables():
	"""Read the shared simulation tables, falling back to the built-in ones."""
	try:
		import yaml

		with open(SIM_TABLES_PATH) as handle:
			tables = yaml.safe_load(handle)
		if not tables or 'cosmology' not in tables:
			raise ValueError(f'{SIM_TABLES_PATH} has no cosmology table')
		return tables
	except (OSError, ImportError, ValueError) as exc:
		logging.getLogger(__name__).warning(
			'Could not read shared simulation tables (%s); using built-in defaults. '
			'Regenerate them with "python -m pipeline.registry" in IA_z_evolution.', exc
		)
		return {
			'cosmology': _FALLBACK_COSMOLOGY,
			'redshifts': _FALLBACK_REDSHIFTS,
			'pi_max': _FALLBACK_PI_MAX,
			'colour_cuts': {},
		}


def get_cosmology_configs():
	"""Return cosmology configurations for each simulation."""
	return _load_sim_tables()['cosmology']


def get_redshift_maps():
	"""Return redshift mappings for each simulation.

	Returns:
		dict: Snapshot to redshift mapping for each simulation
	"""
	return _load_sim_tables()['redshifts']


def get_pi_max_map():
	"""Return the pi_max [Mpc/h] used when projecting the model, per simulation."""
	return _load_sim_tables()['pi_max']


def get_colour_cuts():
	"""Return the redshift-dependent colour cuts keyed by snapshot group name."""
	return _load_sim_tables().get('colour_cuts', {})


def get_data_configs(
		pos_sample='nstar_gt50',
		shape_sample='mstar_gt9p27_mDM_gt11p34',
		probe='DM',
		file_name='IA_data_measurement_z_evolution_',
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
	path_to_h5py = input_path / f'{file_name}{probe}.hdf5'
	if not path_to_h5py.exists():
		path_to_h5py = input_path / f'{file_name}{probe}.hdf5'
	if not path_to_h5py.exists():
		raise FileNotFoundError(
			f'Neither {file_name}{probe}.hdf5 found in {input_path}')

	# Create descriptive output directory name with sample info. A projection-dependent
	# shape sample is a template; render it so the path holds a real name and not "{proj}".
	sample_tag = (
		f'{render_sample(pos_sample, REFERENCE_PROJECTION)}'
		f'_{render_sample(shape_sample, REFERENCE_PROJECTION)}'
	)
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
			p_string_split[2 * idx_proj + 1] = p_string_split[2 * idx_proj + 1][:2] + str(colibre_color_cuts[snapshot])

		p_string = '_ri_'.join(p_string_split)

	return p_string


#: Projection whose rendering of a sample name labels quantities that span both lines
#: of sight (the full covariance, output directories). Matches
#: `reference_projection` in the IA_z_evolution pipeline configs.
REFERENCE_PROJECTION = 'z'

#: Line of sight used for each projection index.
PROJECTION_ORDER = ('y', 'z')


def render_sample(template, projection):
	"""Render a sample name for one projection.

	Aperture shape measurements name the shape sample after the axis it was projected
	along, so `..._projected_aperture_100kpc_reduced_projy` belongs with `LOSy` and
	`..._projz` with `LOSz`. Such samples are passed in as a template containing
	`{proj}`. Names without the placeholder are projection independent and render
	unchanged, which is how every older sample behaves.
	"""
	if template is None or '{proj}' not in template:
		return template
	return template.replace('{proj}', projection)


def extract_snapshot_data(
		measurement_dict,
		cosmo_dict,
		colibre_color_cuts,
		sim,
		snapshot,
		g_string,
		p_string,
		n_projection: int = 2,
		n_projection_multipoles: int = None,
		covariance: str = 'auto',
):
	"""Pull one snapshot's data vectors and covariance out of the measurement dict.

	Args:
		p_string: Shape sample name, optionally containing `{proj}` (see `render_sample`).
		n_projection: Lines of sight combined for the projected correlations (w).
		n_projection_multipoles: Lines of sight combined for the multipoles. Defaults to
			`n_projection`. Usually 1: the real-space multipole clustering has no
			line-of-sight dependence, so combining two projections repeats the same
			numbers and gives a singular covariance.
		covariance: 'full' to require the joint covariance stored in the `w` /
			`multipoles` groups, 'block_diagonal' to force the clustering and alignment
			covariances to be combined with zero cross terms, or 'auto' (default) to use
			the full one when it is present. Which one was used is logged: the two give
			different fits, and it used to depend silently on what happened to be in the
			file.
	"""
	# Define data strings
	# ------------------------------------------------
	# Exchange colour cut with redshift dependent value if present in p_string
	p_string = correct_color_cut_in_data_str(
		p_string=p_string,
		colibre_color_cuts=colibre_color_cuts,
		snapshot=snapshot,
		n_projection=n_projection
	)

	logger = logging.getLogger(__name__)

	if n_projection_multipoles is None:
		n_projection_multipoles = n_projection

	def data_strings(n_proj):
		"""Build the (clustering, alignment) dataset names for a number of projections."""
		if n_proj == 2:
			first, second = PROJECTION_ORDER
			gg = (
				f'D_{g_string}_S_{render_sample(g_string, first)}_LOS{first}'
				f'_D_{g_string}_S_{render_sample(g_string, second)}_LOS{second}'
			)
			gplus = (
				f'D_{g_string}_S_{render_sample(p_string, first)}_LOS{first}'
				f'_D_{g_string}_S_{render_sample(p_string, second)}_LOS{second}'
			)
			return gg, gplus
		if n_proj == 1:
			only = REFERENCE_PROJECTION
			return (
				f'D_{g_string}_S_{render_sample(g_string, only)}_LOS{only}',
				f'D_{g_string}_S_{render_sample(p_string, only)}_LOS{only}',
			)
		raise ValueError(f'Invalid number of projections: {n_proj}. Must be 1 or 2.')

	# w and multipoles may combine a different number of lines of sight.
	gg_string, gplus_string = data_strings(n_projection)
	xi_gg_string, xi_gplus_string = data_strings(n_projection_multipoles)

	# The joint covariance spans both projections, so it is filed under the sample name
	# rendered with the reference projection.
	cov_string = (
		f'D_{render_sample(g_string, REFERENCE_PROJECTION)}'
		f'_S_{render_sample(p_string, REFERENCE_PROJECTION)}_cov'
	)
	# ------------------------------------------------

	if covariance not in ('auto', 'full', 'block_diagonal'):
		raise ValueError(
			f"Invalid covariance option {covariance!r}. "
			f"Choose 'auto', 'full' or 'block_diagonal'."
		)

	has_full = (
		'w' in measurement_dict[sim][snapshot]
		and cov_string in measurement_dict[sim][snapshot]['w']
	)
	if covariance == 'full' and not has_full:
		raise KeyError(
			f'covariance="full" was requested but {cov_string} is not in the "w" group '
			f'for {sim} {snapshot}. Run the giant_cov stage of the IA_z_evolution '
			f'pipeline, or pass covariance="block_diagonal".'
		)

	use_full = has_full and covariance != 'block_diagonal'
	if use_full:
		logger.info('%s %s: using the full covariance %s', sim, snapshot, cov_string)
		w_cov_data = measurement_dict[sim][snapshot]['w'][cov_string] / cosmo_dict[sim]['h'] ** 2
		xi_cov_data = measurement_dict[sim][snapshot]['multipoles'][cov_string]
	else:
		logger.info(
			'%s %s: using block-diagonal covariance (clustering x alignment cross terms '
			'set to zero)', sim, snapshot
		)
		w_gg_cov = measurement_dict[sim][snapshot]['w_gg'][gg_string + '_cov'] / cosmo_dict[sim]['h'] ** 2
		w_gplus_cov = measurement_dict[sim][snapshot]['w_g_plus'][gplus_string + '_cov'] / cosmo_dict[sim]['h'] ** 2

		xi_gg_cov = measurement_dict[sim][snapshot]['multipoles_gg'][xi_gg_string + '_cov']
		xi_gplus_cov = measurement_dict[sim][snapshot]['multipoles_g_plus'][xi_gplus_string + '_cov']

		# Combine into full covariance matrix
		w_cov_data = np.zeros((w_gg_cov.shape[0] + w_gplus_cov.shape[0], w_gg_cov.shape[1] + w_gplus_cov.shape[1]))
		w_cov_data[:w_gg_cov.shape[0], :w_gg_cov.shape[1]] = w_gg_cov
		w_cov_data[w_gg_cov.shape[0]:, w_gg_cov.shape[1]:] = w_gplus_cov

		xi_cov_data = np.zeros((xi_gg_cov.shape[0] + xi_gplus_cov.shape[0], xi_gg_cov.shape[1] + xi_gplus_cov.shape[1]))
		xi_cov_data[:xi_gg_cov.shape[0], :xi_gg_cov.shape[1]] = xi_gg_cov
		xi_cov_data[xi_gg_cov.shape[0]:, xi_gg_cov.shape[1]:] = xi_gplus_cov

	# Load data
	# ------------------------------------------------
	output_dict = {
		# projections
		'rp_gg_data': measurement_dict[sim][snapshot]['w_gg'][gg_string + '_rp'] / cosmo_dict[sim]['h'],
		'w_gg_data': measurement_dict[sim][snapshot]['w_gg'][gg_string] / cosmo_dict[sim]['h'],
		'rp_gplus_data': measurement_dict[sim][snapshot]['w_g_plus'][gplus_string + '_rp'] / cosmo_dict[sim]['h'],
		'w_gplus_data': measurement_dict[sim][snapshot]['w_g_plus'][gplus_string] / cosmo_dict[sim]['h'],

		# multipoles
		'r_gg_data': measurement_dict[sim][snapshot]['multipoles_gg'][xi_gg_string + '_r'] / cosmo_dict[sim]['h'],
		'xi_gg_data': measurement_dict[sim][snapshot]['multipoles_gg'][xi_gg_string],
		'r_gplus_data': measurement_dict[sim][snapshot]['multipoles_g_plus'][xi_gplus_string + '_r'] / cosmo_dict[sim][
			'h'],
		'xi_gplus_data': measurement_dict[sim][snapshot]['multipoles_g_plus'][xi_gplus_string],

		# covariance matrix
		'w_cov_data': w_cov_data,
		'xi_cov_data': xi_cov_data,
	}
	# ------------------------------------------------

	# The covariance is filed under a projection-independent name, so nothing in the key
	# itself says how many projections it spans. Check it against the data vector rather
	# than discovering the mismatch as a nonsensical chi2.
	for cov_key, gg_key, gplus_key in (
			('w_cov_data', 'w_gg_data', 'w_gplus_data'),
			('xi_cov_data', 'xi_gg_data', 'xi_gplus_data'),
	):
		expected = len(output_dict[gg_key]) + len(output_dict[gplus_key])
		actual = output_dict[cov_key].shape[0]
		if actual != expected:
			raise ValueError(
				f'{sim} {snapshot}: {cov_key} is {actual}x{actual} but the data vector has '
				f'{expected} entries (n_projection={n_projection}). The covariance in the '
				f'catalogue was most likely built for a different number of projections.'
			)

	# Set values outside fitting range to zero, TODO: REMOVE AFTER TESTING
	# test_fitting_range = [6/cosmo_dict[sim]['h'], 50/cosmo_dict[sim]['h']]
	# output_dict['w_gg_data'][(output_dict['rp_gg_data'] < test_fitting_range[0]) | (output_dict['rp_gg_data'] > test_fitting_range[1])] = 0
	# output_dict['w_gplus_data'][(output_dict['rp_gplus_data'] < test_fitting_range[0]) | (output_dict['rp_gplus_data'] > test_fitting_range[1])] = 0
	# output_dict['xi_gg_data'][(output_dict['r_gg_data'] < test_fitting_range[0]) | (output_dict['r_gg_data'] > test_fitting_range[1])] = 0
	# output_dict['xi_gplus_data'][(output_dict['r_gplus_data'] < test_fitting_range[0]) | (output_dict['r_gplus_data'] > test_fitting_range[1])] = 0

	return output_dict


def analyze_snapshot(
		sim,
		snapshot,
		redshift,
		data,
		cosmo_dict,
		k_input,
		rp,
		pi_max,
		fitting_range,
		outpath,
		logger,
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
		fitting_range: list = None,
		rp: np.ndarray = None,
		covariance: str = 'auto',
		n_projection_multipoles: int = None,
):
	"""Iterate over simulations and snapshots, analyze each snapshot, and collect results."""
	# Initialize results storage
	output_df_list = []

	def projections_for(spec, sim, default):
		"""Resolve a projection count that may be given per simulation.

		The number of lines of sight is a property of (sample, simulation): a campaign can
		combine two projections for one simulation and a single one for another. Accepting
		a mapping here keeps one call -- and so one output CSV -- per sample. Splitting a
		sample across calls instead would make them collide on `output_csv`, with the
		second silently overwriting the first.
		"""
		if isinstance(spec, dict):
			return spec.get(sim, default)
		return default if spec is None else spec

	for sim in sims:

		h = cosmo_dict[sim]['h']
		rmin = 0.1 / h  # [Mpc]
		rmax = 50. / h  # [Mpc]
		num_r = 50

		# rp is in Mpc and so depends on h: it has to be rebuilt for each simulation.
		# Assigning to `rp` here would freeze the first simulation's grid and silently
		# reuse it for the rest, which matters as soon as one call spans simulations
		# with different h (e.g. TNG300 at 0.6774 after COLIBRE at 0.681). An rp passed
		# in by the caller still overrides it.
		rp_sim = rp if rp is not None else np.geomspace(rmin, rmax, num_r)
		if fitting_range is None:
			fitting_range = [6, 50]
		# Convert to Mpc
		fitting_range_noh = [fitting_range[0] / h, fitting_range[1] / h]

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
					n_projection=projections_for(n_projection, sim, 2),
					n_projection_multipoles=projections_for(
						n_projection_multipoles, sim, projections_for(n_projection, sim, 2)
					),
					covariance=covariance,
				)
			except KeyError as e:
				logger.warning(f'KeyError occurred: {e}')
				logger.warning('Skipping to next snapshot.')
				continue

			(
				best_fit_params_projections, posterior_std_projections, reduced_chi2_projections,
				best_fit_params_multipoles, posterior_std_multipoles, reduced_chi2_multipoles
			) = analyze_snapshot(
				sim, snapshot, redshift, data, cosmo_dict, k_input, rp_sim,
				pi_max[sim] / h, fitting_range_noh, data_config['outpath'], logger
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
