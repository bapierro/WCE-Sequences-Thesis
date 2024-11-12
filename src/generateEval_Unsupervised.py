from wce_clustering_v2 import WCECluster
from model_name import Model
import pandas as pd
import os
from datetime import datetime


base_path = "../evaluation_data/AnnotatedVideos_30FPS"
folders = [f.path for f in os.scandir(base_path) if f.is_dir()]

from wce_clustering_v2 import WCECluster
from model_name import Model
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil

# Function to generate a unique run ID based on the current timestamp
def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_metrics(csv_file, metrics, output_dir, dir_name):
    """
    Generates boxplots for specified metrics across models and saves them in both PNG and SVG formats.

    :param csv_file: Path to the CSV file containing the data.
    :param metrics: A list of performance metrics to plot.
    :param output_dir: Directory to save the generated plots.
    :param dir_name: Name of the directory (used for labeling the plot file).
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Adjust the model names in the column headers to match those in the CSV
    df.columns = df.columns.str.replace('CENDO_FM', 'CendoFM')
    df.columns = df.columns.str.replace('ENDO_FM', 'EndoFM')
    df.columns = df.columns.str.replace('RES_NET_101', 'ResNet101')
    df.columns = df.columns.str.replace('RES_NET_50', 'ResNet50')
    df.columns = df.columns.str.replace('Swin_v2_B', 'SwinV2-B')

    # List of models to include in the plots
    models = ['CendoFM', 'EndoFM','SwinV2-B','ResNet50']
    
    metric_name_mapping = {
        'Calinski–Harabasz Index': 'CHI',
        'Davies-Bouldin Index': 'DBI',
        # 'Density-Based Clustering Validation Index':'DBCVI',
        "nmi":"NMI",
        "ari":"ARI",
        'Silhouette Coefficient': "Silhouette"
        
    }

    # Determine the number of rows and columns for subplots
    max_cols = 3
    num_metrics = len(metrics)
    num_cols = min(num_metrics, max_cols)
    num_rows = (num_metrics + max_cols - 1) // max_cols  # Ceiling division

    # Initialize the plot with subplots for each metric
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14,22))
    # fig.suptitle(f'Evaluation', fontsize=16)

    # Flatten axes array for easy iteration
    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable
    elif num_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Prepare the list of columns for this metric
        metric_columns = [f'{metric}_{model}' for model in models]

        # Check if all columns exist in the DataFrame
        existing_columns = [col for col in metric_columns if col in df.columns]

        if not existing_columns:
            print(f"No data found for metric '{metric}'. Skipping.")
            # Remove axis if no data to plot
            fig.delaxes(ax)
            continue

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
        sns.violinplot(
            x='Model',
            y=metric,
            data=melted_data,
            palette='crest',
            ax=ax,
            inner='quart',
            density_norm='width',
            hue='Model'
        )
        sns.stripplot(
            x='Model',
            y=metric,
            data=melted_data,
            color='black',
            alpha=0.7,
            jitter=False,
            ax=ax,
            size=5
        )

        metric_key = metric.split("_")[0]
        metric_title = metric_name_mapping.get(metric_key, metric_key.title())
        # Customize the subplot
        # metric_title = metric.split("_")[0].title()
        # metric_title = metric_title.upper() if len(metric_title) == 3 else metric_title
        # ax.set_title(metric_title)
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel(metric_title)

        
        
        # Adjust the spine linewidths to make axes thinner
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)  # Set to desired thinness

        
        # Ensure that ticks are visible and appropriately sized
        ax.tick_params(axis='both', which='both', length=5, width=0.5,labelsize=20)
        
        ax.grid(True, axis='y', linestyle='-', alpha=0.7)
        
        if metric_key in ['nmi', 'ari', 'accuracy','silhouette']:
        # Get current y-axis limits
            # ymin, ymax = ax.get_ylim()
            # Set y-axis ticks to be between 0 and 1
            ax.set_yticks([tick for tick in ax.get_yticks() if tick <= 1.1])
        


    # Hide any unused subplots
    total_subplots = num_rows * num_cols
    if total_subplots > num_metrics:
        for j in range(num_metrics, total_subplots):
            fig.delaxes(axes[j])

    # Improve layout spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96],w_pad=2)  # Leave space for the main title
    # plt.subplots_adjust(wspace=0.6, hspace=0.6)  # Adjust space between plots

    # Do not remove the spines and ticks
    sns.despine(offset=10, trim=True)

    # Adjust layout and save the figure
    os.makedirs(output_dir, exist_ok=True)
    plot_file_png = os.path.join(output_dir, f"{dir_name}_performance_metrics.png")
    plt.savefig(plot_file_png, dpi=300)
    print(f"Plot saved to {plot_file_png}")
    plot_file_svg = os.path.join(output_dir, f"{dir_name}_performance_metrics.svg")
    plt.savefig(plot_file_svg, format='svg', dpi=300)
    print(f"Plot saved to {plot_file_svg}")
    plt.close(fig)
    
    
def process_directories(root_dir, metrics, output_dir):
    """
    Process each directory within the root directory and plot metrics from CSV files.

    :param root_dir: The root directory containing subdirectories with CSV files.
    :param metrics: List of metrics to include in the plots.
    :param output_dir: Directory to save all plots.
    """
    for file_name in os.listdir(root_dir):
        if file_name.endswith('.csv'):
            csv_path = os.path.join(root_dir, file_name)
            print(f"Processing file: {csv_path}")
            
            # Extract directory name without the run_id for labeling
            dir_name = os.path.basename(root_dir)
            
            # Plot metrics for the CSV file
            plot_metrics(csv_path, metrics, output_dir, dir_name)

def collect_best_performances(experiments_root, metrics, best_metric='accuracy_with_anomalies'):
    """
    Collects the best performance across different minCL and Sigma values for each model,
    and returns a DataFrame with one column per model, including all specified metrics.

    :param experiments_root: Path to the root directory containing experiment result directories.
    :param metrics: A list of performance metrics to include.
    :param best_metric: The metric used to select the best performance per model.
    :return: DataFrame containing the best performances, with models as columns.
    """
    all_data = []
    # Iterate through Sigma directories
    for sigma_dir in os.listdir(experiments_root):
        sigma_path = os.path.join(experiments_root, sigma_dir)  # Corrected path
        if os.path.isdir(sigma_path):
            # Iterate through CSV files in the Sigma directory
            for csv_file in os.listdir(sigma_path):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(sigma_path, csv_file)

                    # Read the CSV file
                    df = pd.read_csv(csv_path)

                    # Extract model name from file name if 'Model' column not in df
                    if 'Model' not in df.columns:
                        # Expected filename format: 'combined_{Model}_{Sigma}_results.csv'
                        model_name = csv_file.replace('combined_', '').replace('_results.csv', '')
                        # Remove '_SigmaX' from model_name
                        model_name = model_name.replace(f'_{sigma_dir}', '')
                        df['Model'] = model_name

                    # Append to the list
                    all_data.append(df)

    if not all_data:
        raise ValueError("No data files were found in the given experiments_root.")

    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Check if the metrics exist in the DataFrame
    for metric in metrics:
        if metric not in combined_df.columns:
            raise ValueError(f"The specified metric '{metric}' is not found in the data.")

    # Check if the best_metric exists in the DataFrame
    if best_metric not in combined_df.columns:
        raise ValueError(f"The specified best_metric '{best_metric}' is not found in the data.")

    # Group by 'Dataset' if it exists, otherwise proceed without it
    if 'Dataset' in combined_df.columns:
        group_columns = ['Dataset']
    else:
        group_columns = []

    # Initialize a list to store results
    results = []

    # For each group (Dataset or overall), find the best performance per model
    for _, group_df in combined_df.groupby(group_columns):
        # Sort the DataFrame by 'best_metric' in descending order
        group_df_sorted = group_df.sort_values(by=[best_metric], ascending=False)
        # Drop duplicates to keep the best performance per model
        best_performance_per_model = group_df_sorted.drop_duplicates(subset=['Model'], keep='first')

        # Extract 'Sigma' and 'minCl' for each model
        sigma_minCl_df = best_performance_per_model[['Model', 'Sigma', 'minCl']].set_index('Model')

        # Pivot the DataFrame to have models as columns (excluding 'Sigma' and 'minCl')
        pivot_df = best_performance_per_model.pivot(
            index=group_columns if group_columns else None,
            columns='Model',
            values=metrics
        )

        # Flatten the MultiIndex columns
        pivot_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in pivot_df.columns]

        # Add 'Sigma' and 'minCl' back to the DataFrame
        for model in sigma_minCl_df.index:
            pivot_df[f'Sigma_{model}'] = sigma_minCl_df.loc[model, 'Sigma']
            pivot_df[f'minCl_{model}'] = sigma_minCl_df.loc[model, 'minCl']

        # Reset index to turn index into columns
        pivot_df = pivot_df.reset_index()

        # Append to results
        results.append(pivot_df)

    # Concatenate all results
    if results:
        final_df = pd.concat(results, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df

def merge_model_evaluation_csvs(base_path):
    """
    Merge all CSV files within each model's Sigma directory across all datasets,
    aggregating per sigma and keeping models separate.

    :param base_path: Path to the Evaluation directory containing dataset directories.
    :return: Dictionary with (model, sigma) tuples as keys and concatenated DataFrames as values.
    """
    evaluation_data = {}

    # Iterate through each dataset directory
    for dataset_dir in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_dir)

        if os.path.isdir(dataset_path):  # Ensure it's a directory (e.g., ncm_12)
            # Inside dataset directory, look for model directories (e.g., resnet, sigma1)
            for model_dir in os.listdir(dataset_path):
                model_path = os.path.join(dataset_path, model_dir)

                if os.path.isdir(model_path):  # Ensure it's a directory (model directory)
                    # Look for subdirectories starting with 'Sigma'
                    sigma_dirs = [d for d in os.listdir(model_path) if d.startswith('Sigma')]

                    for sigma_dir in sigma_dirs:
                        sigma_path = os.path.join(model_path, sigma_dir)

                        if os.path.isdir(sigma_path):
                            # Initialize list for this model and sigma
                            key = (model_dir, sigma_dir)
                            if key not in evaluation_data:
                                evaluation_data[key] = []

                            # Inside Sigma directory, look for CSV files
                            csv_files = [f for f in os.listdir(sigma_path) if f.endswith('.csv')]

                            for csv_file in csv_files:
                                file_path = os.path.join(sigma_path, csv_file)

                                # Read the CSV file
                                df = pd.read_csv(file_path)

                                # Add columns for 'Dataset', 'Model', 'Sigma'
                                df['Dataset'] = dataset_dir
                                df['Model'] = model_dir
                                df['Sigma'] = sigma_dir

                                # Store the data in the evaluation_data dictionary under the (model, sigma) key
                                evaluation_data[key].append(df)

    # Concatenate all the CSV files for each (model, sigma)
    combined_data = {}
    for key, dfs in evaluation_data.items():
        if dfs:  # Ensure there are DataFrames to concatenate
            combined_data[key] = pd.concat(dfs, ignore_index=True)
        else:
            combined_data[key] = pd.DataFrame()

    return combined_data

def cleanup_previous_runs(base_dir, keep_recent=5):
    """
    Remove older run directories, keeping only the most recent ones.

    :param base_dir: The base directory containing run directories.
    :param keep_recent: Number of recent runs to keep.
    """
    runs = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
        reverse=True
    )
    for run in runs[keep_recent:]:
        run_path = os.path.join(base_dir, run)
        try:
            shutil.rmtree(run_path)
            print(f"Removed old run directory: {run_path}")
        except Exception as e:
            print(f"Error removing {run_path}: {e}")

if __name__ == "__main__":
    # Generate a unique run ID
    run_id = generate_run_id()
    print(f"Starting run with ID: {run_id}")

    # Define base directories for outputs
    dumps_root = os.path.join("./dumps", "DownsampledVideos", run_id)
    evaluation_unsupervised_path = os.path.join(dumps_root, "Evaluation_Unsupervised")
    os.makedirs(evaluation_unsupervised_path, exist_ok=True)

    # Optional: Clean up older runs to keep the dumps directory manageable
    # Uncomment the following line to activate cleanup
    # cleanup_previous_runs(os.path.join("./dumps", "subsection"), keep_recent=5)

    # Iterate through folders and apply clustering
    for folder in folders:
        print("-----------------------")
        print(f"Section: {os.path.basename(folder)}")

        WCECluster(
            dataset_path=folder,
            batch_size=64,
            smooth=True,
            fps=1,
            recompute=True,
            backbones=[Model.CENDO_FM,Model.ENDO_FM,Model.Swin_v2_B,Model.RES_NET_50],
            evaluate=True,
            save_representatives=True,
            sigmas=[5],
            external_validation=False,
            run_id=run_id,  
            output_root=dumps_root 
        ).apply()

    # Merge evaluation CSVs
    merged_results = merge_model_evaluation_csvs(evaluation_unsupervised_path)

    # Define metrics to include
    metrics_to_include = [
        "Davies-Bouldin Index",
        'Calinski–Harabasz Index',
        "Silhouette Coefficient"
        ]

    best_metric = "Calinski–Harabasz Index"

    # Define experiments root with run_id
    experiments_root = os.path.join("./Experiments", "UnsupervisedEvaluation", run_id)
    os.makedirs(experiments_root, exist_ok=True)

    # Save the combined results into separate files for each model and sigma
    for (model, sigma), combined_df in merged_results.items():
        if combined_df.empty:
            print(f"No data for model {model} and sigma {sigma}. Skipping.")
            continue

        # Clean the sigma name to remove any file system incompatible characters if necessary
        safe_sigma = sigma.replace('/', '_').replace('\\', '_')
        # Create directory for this Sigma within 'Experiments/UnsupervisedEvaluation/{run_id}/'
        sigma_dir = os.path.join(experiments_root, safe_sigma)
        os.makedirs(sigma_dir, exist_ok=True)
        # Build the output path
        output_path = os.path.join(sigma_dir, f"combined_{model}_{safe_sigma}_results.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved combined results to {output_path}")

    # Collect best performances
    best_results_df = collect_best_performances(experiments_root, metrics=metrics_to_include, best_metric=best_metric)

    # Define output directories for best performances
    bestperformances_root = os.path.join("BestPerformances", run_id)
    os.makedirs(bestperformances_root, exist_ok=True)

    # Define output CSV and LaTeX filenames
    safe_metric = best_metric.replace('/', '_').replace('\\', '_')
    output_csv_path = os.path.join(bestperformances_root, f"best_performances_{safe_metric}.csv")
    output_tex_file = os.path.join(bestperformances_root, f"performance_comparison_table_{safe_metric}.tex")

    # Save the best performances to a CSV file
    try:
        best_results_df.to_csv(output_csv_path, index=False)
        print(f"The best performances have been saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV for {best_metric}: {e}")

    # Define output plots directory
    output_plots_dir = os.path.join("BestPerformances_Plots", run_id)
    os.makedirs(output_plots_dir, exist_ok=True)

    # Plot metrics boxplots
    process_directories(bestperformances_root, metrics_to_include, output_plots_dir)