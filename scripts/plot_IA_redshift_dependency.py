import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Save output dataframe to csv
    parent = Path('/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/run_20260210')
    addendum = 'stars_nstar_gt50_vsig_lt1'
    input_df = pd.read_csv(parent / f'IA_fitting_results_summary_{addendum}.csv', sep='\t')
    plot_savepath = parent / 'plots'
    plot_savepath.mkdir(parents=True, exist_ok=True)

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

    plt.suptitle(f'Run 20260210, {addendum}')
    plt.tight_layout()
    plt.savefig(plot_savepath / f'IA_parameters_vs_redshift_{addendum}.png')
    plt.close()

if __name__ == "__main__":
    main()