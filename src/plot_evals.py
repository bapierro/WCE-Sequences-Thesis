import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_boxplots(csv_file, metrics):
    """
    Generates boxplots for specified metrics across models.

    :param csv_file: Path to the CSV file containing the data.
    :param metrics: A list of performance metrics to plot.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Adjust the model names in the column headers to match those in the CSV
    df.columns = df.columns.str.replace('CENDO_FM', 'CendoFM')
    df.columns = df.columns.str.replace('ENDO_FM', 'EndoFM')
    df.columns = df.columns.str.replace('RES_NET_101', 'ResNet101')

    # List of models to include in the plots
    models = ['CendoFM', 'EndoFM', 'ResNet101']

    # Loop over each metric to create individual plots
    for metric in metrics:
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

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        # Create the boxplot
        sns.boxplot(x='Model', y=metric, data=melted_data, palette='Set3')

        # Overlay the data points
        sns.stripplot(
            x='Model',
            y=metric,
            data=melted_data,
            color='black',
            alpha=0.5,
            jitter=True
        )

        # Customize the plot
        plt.title(f'Boxplot of {metric} across Models')
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()

# Example usage
metrics_to_include = [
    'accuracy_with_anomalies',
    'nmi_with_anomalies',
    'ari_with_anomalies'
]

if __name__ == '__main__':
    plot_metrics_boxplots(
        "BestPerformances/heavyAugsOverlap4_Views/best_performances_accuracy_with_anomalies.csv",
        metrics_to_include
    )