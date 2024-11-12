
import os
import pandas as pd
from model_name import Model
from wce_clustering_v2 import WCECluster
import matplotlib.pyplot as plt
import seaborn as sns
# Path to your selection folder

# Now plot the boxplots
def plot_metrics_boxplots(df, metrics, output_root, name):
    """
    Generate and save boxplots with overlaid data points for the specified metrics across all models.

    Parameters:
    - df: DataFrame containing the metrics.
    - metrics: List of metrics to plot.
    - output_root: Root directory for outputs.
    - name: Name of the experiment.
    """
    # Set Seaborn style to white for a clean look
    sns.set_style("white")

    df['model_name'] = df['model_name'].replace({
        'ENDO_FM': 'Endo-FM',
        'CENDO_FM': 'Cendo-FM',
        'RES_NET_101': 'ResNet-101'
    })  
    df['Model'] = df['model_name']
    df['Sigma'] = df['sigma']

    # Exclude random segmentation if not interesting
    df = df[df['method'] != 'Random_Segmentation']
    # Loop over each metric to create individual plots
    for metric in metrics:
        # Check if the metric exists in the DataFrame
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in the data. Skipping.")
            continue

        # Initialize the plot
        plt.figure(figsize=(8, 6))

        # Create the boxplot with slightly grey fill and black outlines
        sns.boxplot(
            x='Model',
            y=metric,
            data=df,
            width=0.4,
            showfliers=False,
            # boxprops=dict(facecolor='#D3D3D3'),
            # medianprops=dict(color='black'),
            # whiskerprops=dict(color='black'),
            # capprops=dict(color='black'),
            # flierprops=dict(markeredgecolor='black', marker='o', markersize=5)
        )

        # Overlay the data points with stripplot in black
        sns.stripplot(
            x='Model',
            y=metric,
            data=df,
            color='black',
            alpha=0.8,
            jitter=True,
            size=5
        )

        # Customize the plot
        plt.xlabel('Backbone', fontsize=14, fontweight='bold')

        # Conditional y-axis label formatting
        # metric_base = metric.split("_")[0]
        # y_label = metric_base.upper() if len(metric_base) == 3 else metric_base.replace('_', ' ').title()
        plt.ylabel(metric, fontsize=14, fontweight='bold')

        # Access the current Axes instance
        ax = plt.gca()

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        # Set the linewidth of the bottom and left spines
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)

        # Ensure that only bottom and left ticks are present
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Enhance tick label visibility
        ax.tick_params(axis='x', labelsize=6, width=1.2, length=5)
        ax.tick_params(axis='y', labelsize=6, width=1.2, length=5)
        
        
        # plt.grid(True)
        ax.yaxis.grid(True)

        # Determine the maximum and minimum values in the data for the current metric
        max_val = df[metric].max()
        min_val = df[metric].min()

        # Set y-axis limits: start at 0 or the minimum value, end slightly above the max value
        upper_limit = max_val * 1.05
        lower_limit = min_val * 0.95
        # plt.ylim(bottom=min(0, min_val), top=upper_limit)

        # Enhance tick label visibility
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Tight layout for better spacing
        plt.tight_layout()

        # Save the plot
        plots_dir = os.path.join(output_root, "Plots", name)
        os.makedirs(plots_dir, exist_ok=True)
        new_name = str(metric).replace(" ","_")
        plot_file = os.path.join(plots_dir, f"{new_name}_boxplot.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot for metric '{metric}' saved to {plot_file}")
        
if __name__ == "__main__":
    path = "../evaluation_data/AnnotatedVideos_30FPS/"
    folders = [f.path for f in os.scandir(path) if f.is_dir()]

    # Global list to collect all metrics
    all_metrics = []

    for folder in folders:
        print("-----------------------")
        print(f"Processing dataset: {os.path.basename(folder)}")
        clusterer = WCECluster(
            dataset_path=folder,
            minCl=[300],
            sigmas=[4],
            batch_size=32,
            smooth=True,
            fps=30,
            save_representatives=False,  # Set to False since we'll plot later
            recompute=False,
            backbones=[Model.CENDO_FM, Model.ENDO_FM, Model.RES_NET_101],
            evaluate=True
        )
        clusterer.apply()

        # Collect metrics from this dataset
        all_metrics.extend(clusterer.all_metrics)

    # After processing all datasets, create a DataFrame
    combined_metrics_df = pd.DataFrame(all_metrics)
    print("Combined metrics collected over all datasets.")

    # Save the combined metrics to a CSV file
    merged_output_dir = os.path.join('./dumps', "Merged_Evaluation")
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_output_csv = os.path.join(merged_output_dir, "all_datasets_merged_metrics.csv")
    combined_metrics_df.to_csv(merged_output_csv, index=False)
    print(f"Merged evaluation metrics saved to {merged_output_csv}")

    

    # Define the metrics you want to plot
    metrics_to_plot = [
        # 'Temporal Purity',
        'Davies-Bouldin Index',
        'Calinski–Harabasz Index',
        # 'Intercluster Distance',
        # 'Intracluster Distance',
        'Intracluster Variance',
        # 'Number   of Transitions',
        # 'Average Segment Length'
    ]

    # Call the function to plot the metrics
    plot_metrics_boxplots(
        df=combined_metrics_df,
        metrics=metrics_to_plot,
        output_root='./dumps',
        name='Combined_Datasets'
    )