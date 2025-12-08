import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Save output dataframe to csv
    output_df = pd.read_csv('/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/IA_fitting_results_summary.csv', sep='\t')

    # Create plots of A_IA and b_g vs redshift
    for estimator in output_df['estimator'].unique():
        df_estimator = output_df[output_df['estimator'] == estimator]

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        for idx, param in enumerate(['A_IA', 'b_g']):
            for sim in df_estimator['simulation'].unique():
                df_plot = df_estimator[df_estimator['simulation'] == sim]
                ax[idx].errorbar(
                    df_plot['redshift'],
                    df_plot[param],
                    yerr=df_plot[param + '_err'],
                    fmt='o',
                    label=sim,
                )
            ax[idx].set_xlabel('Redshift')
            ax[idx].set_ylabel(param)
            ax[idx].set_title(f'{param} vs Redshift for {estimator}')
            ax[idx].legend()

        plt.tight_layout()
        plt.savefig(f'/home/dneup16/leiden_phd/scripts/results/IA_redshift_dependency_simulations/{estimator}_IA_parameters_vs_redshift.png')
        plt.close()

if __name__ == "__main__":
    main()