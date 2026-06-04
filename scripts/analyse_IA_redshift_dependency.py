# Limit number of cores
import os

from jedi.api import file_name

os.nice(19)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from src_IA_redshift_dependency import *

def main():
	"""Main analysis pipeline for IA redshift dependency."""
	# Setup
	logger = setup_logger()
	cosmo_dict = get_cosmology_configs()
	k_input = np.geomspace(1e-5, 500, 1000)
	fitting_range = [6, 50] # In Mpc/h, will be converted to Mpc in function "iterate_over_simulations_and_snapshots()"
	input_path = '/Users/6918522/Documents/Work/IA_z_evolution/data/IA_correlations/'
	output_path = '/Users/6918522/Documents/Work/IA_z_evolution/data/Modelling_parameters/'
	file_name = "IA_data_measurement_z_evolution_vsig_thresholds_"

	# Define measurement list
	sample_list = [
		["mstar_gt9p27_mDM_gt11p34_vsig_lt0.1", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_gt0.1", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_lt0.5", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_gt0.5", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_lt1", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_gt1", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_lt1.5", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_gt1.5", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_lt2", "nstar_gt50"],
		["mstar_gt9p27_mDM_gt11p34_vsig_gt2", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_ri_gt", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_ri_lt", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_q0", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mlt11", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mgt11", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mlt11", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mgt11", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mlt10", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0_mgt10", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mlt10", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0_mgt10", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_lt1.0", "nstar_gt50"],
		# ["mstar_gt9p27_mDM_gt11p34_vsig_gt1.0", "nstar_gt50"],
	]
	probe_list = ['DM', 'stars']
	sim_list = ['L400_m7'] # 'TNG300'
	n_projection_list = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
	# n_projection_list = [1, 1]

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
				file_name=file_name,
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
				fitting_range=fitting_range,
				logger=logger,
			)

if __name__ == '__main__':
	main()