# Limit number of cores
import os

os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse

import yaml

from src_IA_redshift_dependency import *


def apply_sim_aliases(manifest, *tables):
	"""Let variant labels inherit the tables of the simulation they are a variant of.

	A campaign can split one simulation into separately labelled variants -- different
	halo finders, for instance -- and the catalogue is grouped by those labels. Cosmology,
	redshifts and pi_max are only defined for the underlying simulation, so without this
	every lookup for "L200_m7_Subfind" would raise a KeyError.
	"""
	aliases = manifest.get('sim_aliases') or {}
	for label, key in aliases.items():
		for table in tables:
			if label not in table and key in table:
				table[label] = table[key]
	return aliases


def load_manifest(path):
	"""Read a catalogue manifest written by the IA_z_evolution pipeline.

	The manifest lists exactly which samples, simulations and snapshots are present in
	the catalogue, so the sample list does not have to be kept in sync by hand here.
	"""
	with open(path) as handle:
		manifest = yaml.safe_load(handle)
	if not manifest or 'samples' not in manifest:
		raise ValueError(f'{path} is not a catalogue manifest')
	return manifest


def parse_args(argv=None):
	parser = argparse.ArgumentParser(
		description='Fit the IA redshift dependence of a measurement catalogue.'
	)
	parser.add_argument(
		'--manifest',
		help=(
			'Catalogue manifest written by the IA_z_evolution pipeline '
			'(…_manifest.yaml). Supplies the sample list, simulations, snapshots and '
			'number of projections. Without it the built-in defaults below are used.'
		),
	)
	parser.add_argument('--sim', action='append', help='Restrict to these simulations.')
	parser.add_argument('--sample', action='append', help='Restrict to shape samples containing this substring.')
	parser.add_argument('--probe', default='stars', choices=['stars', 'DM'])
	parser.add_argument(
		'--n-projection',
		type=int,
		choices=[1, 2],
		help='Override the number of projections for w given by the manifest.',
	)
	parser.add_argument(
		'--n-projection-multipoles',
		type=int,
		choices=[1, 2],
		help=(
			'Override the number of projections for the multipoles. Normally 1: the '
			'real-space multipole clustering has no line-of-sight dependence, so '
			'combining two projections gives a singular covariance.'
		),
	)
	parser.add_argument(
		'--covariance',
		default='auto',
		choices=['auto', 'full', 'block_diagonal'],
		help=(
			'Which covariance to fit with. "full" uses the joint covariance including '
			'the clustering x alignment cross terms; "block_diagonal" zeroes them; '
			'"auto" (default) uses the full one when the catalogue has it.'
		),
	)
	parser.add_argument('--fitting-range', type=float, nargs=2, default=[6, 50], metavar=('RMIN', 'RMAX'))
	parser.add_argument('--output-path', default='/Users/6918522/Documents/Work/IA_z_evolution/data/Modelling_parameters/')
	return parser.parse_args(argv)


def measurements_from_manifest(manifest, args):
	"""Turn a manifest into (input_path, file_name, samples, sims, n_proj, n_proj_multipoles).

	A manifest lists one entry per (sample, projection count), so a sample whose
	simulations use different numbers of projections appears more than once. Those are
	merged back into a single measurement here, carrying the counts as {sim: n} maps.
	One call per sample matters: the output CSV is named after the sample alone, so two
	calls for one sample would overwrite each other's results.
	"""
	merged = {}
	order = []
	for entry in manifest['samples']:
		if args.sample and not any(s in entry['shape_sample'] for s in args.sample):
			continue
		sims = entry['sims']
		if args.sim:
			sims = [s for s in sims if s in args.sim]
		if not sims:
			continue
		n_proj = args.n_projection or entry.get('n_projection', 1)
		n_proj_mp = args.n_projection_multipoles or entry.get('n_projection_multipoles', n_proj)

		key = (entry['shape_sample'], entry['pos_sample'])
		if key not in merged:
			merged[key] = {'sims': [], 'n_projection': {}, 'n_projection_multipoles': {}}
			order.append(key)
		record = merged[key]
		for sim in sims:
			if sim not in record['sims']:
				record['sims'].append(sim)
			record['n_projection'][sim] = n_proj
			record['n_projection_multipoles'][sim] = n_proj_mp

	samples, sim_lists, n_projections, n_projections_multipoles = [], [], [], []
	for key in order:
		record = merged[key]
		samples.append([key[0], key[1]])
		sim_lists.append(record['sims'])
		n_projections.append(record['n_projection'])
		n_projections_multipoles.append(record['n_projection_multipoles'])
	return (
		manifest['input_path'], manifest['file_name_prefix'],
		samples, sim_lists, n_projections, n_projections_multipoles,
	)


def builtin_measurements(args):
	"""The hand-maintained configuration, used when no manifest is given."""
	input_path = '/Users/6918522/Documents/Work/IA_z_evolution/data/IA_correlations/'
	file_name = 'IA_data_measurement_z_evolution_reduced_'
	sample_list = [
		['mstar_gt9p27_mDM_gt11p34_projected_stellar_aperture_10kpc_projz', 'nstar_gt50'],
	]
	sim_lists = [args.sim or ['TNG300'] for _ in sample_list]
	n_projections = [args.n_projection or 1 for _ in sample_list]
	n_projections_multipoles = [
		args.n_projection_multipoles or n for n in n_projections
	]
	return input_path, file_name, sample_list, sim_lists, n_projections, n_projections_multipoles


def main(argv=None):
	"""Main analysis pipeline for IA redshift dependency."""
	args = parse_args(argv)

	logger = setup_logger()
	cosmo_dict = get_cosmology_configs()
	pi_max = get_pi_max_map()
	colibre_color_cuts = get_colour_cuts()
	k_input = np.geomspace(1e-5, 500, 1000)
	fitting_range = list(args.fitting_range)  # [Mpc/h], converted to Mpc downstream

	if args.manifest:
		manifest = load_manifest(args.manifest)
		logger.info(
			'Using manifest %s (campaign %s, config hash %s)',
			args.manifest, manifest.get('campaign'), manifest.get('config_hash'),
		)
		aliases = apply_sim_aliases(manifest, cosmo_dict, pi_max)
		if aliases:
			logger.info('Simulation variants resolved via aliases: %s', aliases)
		(
			input_path, file_name, sample_list, sim_lists,
			n_projections, n_projections_multipoles,
		) = measurements_from_manifest(manifest, args)
	else:
		logger.info('No manifest given; using the built-in sample configuration.')
		(
			input_path, file_name, sample_list, sim_lists,
			n_projections, n_projections_multipoles,
		) = builtin_measurements(args)

	if not sample_list:
		logger.error('No samples selected. Check --sample / --sim against the manifest.')
		return 2

	probe = args.probe
	for sample_idx, (shape_sample, pos_sample) in enumerate(sample_list):
		n_projection = n_projections[sample_idx]
		n_projection_multipoles = n_projections_multipoles[sample_idx]
		sim_list = sim_lists[sample_idx]

		def describe(spec):
			if isinstance(spec, dict):
				return ', '.join(f'{sim}:{n}' for sim, n in sorted(spec.items()))
			return str(spec)

		logger.info(
			f'Processing probe {probe} with position sample "{pos_sample}" and shape sample '
			f'"{shape_sample}" over {sim_list}, using {describe(n_projection)} projection(s) '
			f'for w and {describe(n_projection_multipoles)} for the multipoles.'
		)

		data_config = get_data_configs(
			pos_sample=pos_sample,
			shape_sample=shape_sample,
			probe=probe,
			file_name=file_name,
			input_path=input_path,
			output_path=args.output_path,
		)

		# Redshift maps come from get_data_configs, so variant labels need resolving here
		# too -- is_in_snapshot_dict would otherwise skip every snapshot of a variant.
		if args.manifest:
			apply_sim_aliases(manifest, data_config['redshifts'])

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
			n_projection_multipoles=n_projection_multipoles,
			measurement_dict=measurement_dict,
			cosmo_dict=cosmo_dict,
			colibre_color_cuts=colibre_color_cuts,
			k_input=k_input,
			pi_max=pi_max,
			data_config=data_config,
			fitting_range=fitting_range,
			logger=logger,
			covariance=args.covariance,
		)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
