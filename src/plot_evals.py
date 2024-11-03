import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_boxplots(csv_file, metrics, output_dir, dir_name):
    """
    Generates boxplots for specified metrics across models and saves them in a single figure.

    :param csv_file: Path to the CSV file containing the data.
    :param metrics: A list of performance metrics to plot.
    :param output_dir: Directory to save the generated plot.
    :param dir_name: Name of the directory (used for labeling the plot file).
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Adjust the model names in the column headers to match those in the CSV
    df.columns = df.columns.str.replace('CENDO_FM', 'CendoFM')
    df.columns = df.columns.str.replace('ENDO_FM', 'EndoFM')
    df.columns = df.columns.str.replace('RES_NET_101', 'ResNet101')


    # List of models to include in the plots
    models = ['CendoFM', 'EndoFM', 'ResNet101']

    # Initialize the plot with subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))
    fig.suptitle(f'Metrics {dir_name}', fontsize=16)

    for i, metric in enumerate(metrics):
        # Prepare the list of columns for this metric
        metric_columns = [f'{metric}_{model}' for model in models]

        # Check if all columns exist in the DataFrame
        existing_columns = [col for col in metric_columns if col in df.columns]

        # Create a DataFrame with the relevant columns
        data = df[existing_columns]

        # Melt the DataFrame to long format
        melted_data = pd.melt(
            data,
            value_vars=existing_columns,
            var_name='Model',
            value_name=metric
        )

        # Adjust the 'Model' column to have only the model names
        melted_data['Model'] = melted_data['Model'].str.replace(f'{metric}_', '')
        

        # Plot each metric in its respective subplot
        sns.boxplot(x='Model', y=metric, data=melted_data, palette='crest', ax=axes[i],hue="Model", showfliers=False)
        sns.stripplot(
            x='Model',
            y=metric,
            data=melted_data,
            color='black',
            alpha=0.7,
            jitter=False,
            ax=axes[i]
        )

        metric = metric.split("_")[0].title()
        metric = metric.upper() if len(metric) == 3 else metric
        #Customize the subplot
        axes[i].set_title(metric)
        # axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, axis='y', linestyle='-', alpha=0.7)

    sns.despine(offset=10,trim=True)
    
    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the main title
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, f"{dir_name}_performance_metrics.png")
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {plot_file}")

def process_directories(root_dir, metrics, output_dir):
    """
    Process each directory within the root directory and plot metrics from CSV files.

    :param root_dir: The root directory containing subdirectories with CSV files.
    :param metrics: List of metrics to include in the plots.
    :param output_dir: Directory to save all plots.
    """
    # Iterate through each subdirectory in the root directory
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        
        # Check if it is a directory
        if os.path.isdir(dir_path):
            # Look for CSV files within the directory
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.csv'):
                    csv_path = os.path.join(dir_path, file_name)
                    print(f"Processing file: {csv_path}")
                    
                    # Plot metrics for the CSV file
                    plot_metrics_boxplots(csv_path, metrics, output_dir, dir_name)

# Example usage
metrics_to_include = [
    'accuracy_with_anomalies',
    'nmi_with_anomalies',
    'ari_with_anomalies'
]

if __name__ == '__main__':
    # Path to the main directory containing the subdirectories with CSV files
    bestperformances_root = 'BestPerformances'
    output_plots_dir = 'BestPerformances_Plots'
    process_directories(bestperformances_root, metrics_to_include, output_plots_dir)