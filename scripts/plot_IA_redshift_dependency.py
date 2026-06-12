import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_IA_redshift_dependency(
        parent,
        plot_savepath,
        addendum,
    ):

    input_df = pd.read_csv(parent / f'IA_fitting_results_summary_{addendum}.csv', sep='\t')

    markerstyle = {
        'projections': 'v',
        'multipoles': '^',
    }
    colors = {
        'L400_m7': 'blue',
        'TNG300': 'orange',
    }
    # Create plots of A_IA and b_g vs redshift
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for estimator in input_df['estimator'].unique():
        df_estimator = input_df[input_df['estimator'] == estimator]

        for idx, param in enumerate(['A_IA', 'b_g']):
            for sim in df_estimator['simulation'].unique():
                df_plot = df_estimator[df_estimator['simulation'] == sim]
                ax[idx].errorbar(
                    df_plot['redshift'],
                    df_plot[param],
                    yerr=df_plot[param + '_err'],
                    fmt=markerstyle[estimator],
                    color=colors[sim],
                    label=f'{sim}, {estimator}',
                )
            ax[idx].set_xlabel('Redshift')
            ax[idx].set_ylabel(param)
            ax[idx].set_title(f'{param} vs Redshift')
            ax[idx].legend()

    plt.suptitle(f'{addendum}')
    plt.tight_layout()
    plt.savefig(plot_savepath / f'IA_parameters_vs_redshift_{addendum}.png')
    plt.close()

def main():
    # Save output dataframe to csv
    parent = Path('/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/run_20260311_NL_scaling')

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

    for probe in probe_list:
        for sample in sample_list:
            gstring = sample[1]
            pstring = sample[0]
            addendum = f'{probe}_{gstring}_{pstring}'

            plot_savepath = parent / 'plots'
            plot_savepath.mkdir(parents=True, exist_ok=True)

            plot_IA_redshift_dependency(
                parent=parent,
                plot_savepath=plot_savepath,
                addendum=addendum
            )

if __name__ == "__main__":
    main()