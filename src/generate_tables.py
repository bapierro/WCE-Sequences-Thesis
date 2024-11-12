import os
import pandas as pd
import re

def collect_best_performances(experiments_root, metrics, best_metric='accuracy_with_anomalies'):
    """
    Collects the best performance across different minCL and Sigma values for each model,
    and returns a DataFrame with one column per model, including all specified metrics.

    :param experiments_root: Path to the root directory containing ExperimentsMinCl* directories.
    :param metrics: A list of performance metrics to include.
    :param best_metric: The metric used to select the best performance per model.
    :return: DataFrame containing the best performances, with models as columns.
    """
    all_data = []

    # Iterate through each ExperimentsMinCl* directory
    for mincl_dir in os.listdir(experiments_root):
        mincl_path = os.path.join(experiments_root, mincl_dir)
        if os.path.isdir(mincl_path) and mincl_dir.startswith('ExperimentsMinCl'):
            # Extract minCL value from directory name
            minCL_value = int(mincl_dir.replace('ExperimentsMinCl', ''))

            # Iterate through Sigma directories
            for sigma_dir in os.listdir(mincl_path):
                sigma_path = os.path.join(mincl_path, sigma_dir)
                if os.path.isdir(sigma_path):
                    # Iterate through CSV files in the Sigma directory
                    for csv_file in os.listdir(sigma_path):
                        if csv_file.endswith('.csv'):
                            csv_path = os.path.join(sigma_path, csv_file)

                            # Read the CSV file
                            df = pd.read_csv(csv_path)

                            # Add columns for 'minCL', 'Sigma', 'Model' if not present
                            df['minCL'] = minCL_value
                            df['Sigma'] = sigma_dir

                            # Extract model name from file name if 'Model' column not in df
                            if 'Model' not in df.columns:
                                # Expected filename format: 'combined_{Model}_{Sigma}_results.csv'
                                model_name = csv_file.replace('combined_', '').replace('_results.csv', '')
                                # Remove '_SigmaX' from model_name
                                model_name = model_name.replace(f'_{sigma_dir}', '')
                                df['Model'] = model_name

                            # Append to the list
                            all_data.append(df)

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
        # For each model, find the row with the best performance based on the best_metric
        best_performance_per_model = group_df.loc[group_df.groupby('Model')[best_metric].idxmax()]

        # Pivot the DataFrame to have models as columns
        pivot_df = best_performance_per_model.pivot(
            index=group_columns, columns='Model', values=metrics + ['Sigma', 'minCL']
        )

        # Flatten the MultiIndex columns
        pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

        # Reset index to turn index into columns
        pivot_df = pivot_df.reset_index()

        # Append to results
        results.append(pivot_df)

    # Concatenate all results
    final_df = pd.concat(results, ignore_index=True)

    return final_df

def generate_latex_table(csv_file, metrics):
    """
    Generates LaTeX code for a table comparing EndoFM and ResNet101 across datasets.

    :param csv_file: Path to the CSV file containing the data.
    :param metrics: A list of performance metrics to include in the table.
    :return: LaTeX code as a string.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Adjust the model names in the column headers to match those in the CSV
    df.columns = df.columns.str.replace('ENDO_FM', 'EndoFM')
    df.columns = df.columns.str.replace('CENDO_FM', 'CendoFM')
    df.columns = df.columns.str.replace('RES_NET_101', 'ResNet101')

    # List of models to include in the table
    models = ['EndoFM', 'ResNet101','CendoFM']

    # Adjust dataset names to remove underscores and hyphens (e.g., 'ncm_1' to 'ncm1')
    df['Dataset'] = df['Dataset'].str.replace('_', '').str.replace('-', '')

    # Create a function to extract numerical part for sorting
    def dataset_sort_key(name):
        # Match 'ncm' or 'ncmrmv' followed by numbers
        m = re.match(r'(ncm)(rmv)?(\d+)', name)
        if m:
            prefix = m.group(1)
            rmv = m.group(2)  # 'rmv' or None
            num = int(m.group(3))
            # Assign a number for sorting
            # 'ncm' datasets first, then 'ncmrmv'
            prefix_order = {'ncm': 0, 'ncmrmv': 1}
            rmv_part = 'rmv' if rmv else ''
            prefix_key = prefix + rmv_part
            order = prefix_order.get(prefix_key, 2)
            return (order, num)
        else:
            return (2, name)  # Place unknown datasets at the end

    # Extract datasets
    datasets = df['Dataset'].unique()

    # Sort datasets according to the desired order
    sorted_datasets = sorted(datasets, key=dataset_sort_key)

    # Prepare the LaTeX table header with dynamic metrics
    columns = ' & '.join(['Accuracy', 'NMI', 'ARI', 'Sigma', 'MinCL'])
    latex_table = r'''\begin{table}[h]
\small
\setlength\tabcolsep{3pt}
    \centering
    \vspace{-0.05in}
    \begin{tabular}{cllllll}
    \toprule
        Dataset & Backbone & %s \\
        \midrule
''' % columns

    # For each dataset in sorted order
    for dataset in sorted_datasets:
        # Get the rows corresponding to this dataset
        dataset_rows = df[df['Dataset'] == dataset]
        if dataset_rows.empty:
            continue

        # Add the dataset name in the first column, merged over the number of models
        latex_table += r'    \multirow{%d}{*}{%s}' % (len(models), dataset) + '\n'

        for idx, model in enumerate(models):
            row = dataset_rows.iloc[0]  # Adjusted to get the correct row per model
            metric_values = []
            for metric in metrics:
                # Attempt to retrieve the metric value
                try:
                    value = row[f'{metric}_{model}']
                except KeyError:
                    value = 'N/A'
                if isinstance(value, (int, float)):
                    value_formatted = f'{value:.2f}'
                else:
                    value_formatted = value
                metric_values.append(value_formatted)

            # Process Sigma to extract numerical value
            sigma_value = row[f'Sigma_{model}']
            # Assuming Sigma is like 'Sigma3', extract the number
            sigma_num = re.findall(r'\d+', str(sigma_value))
            if sigma_num:
                sigma_value = float(sigma_num[0])
            else:
                sigma_value = 'N/A'

            minCL_value = row[f'minCL_{model}']

            # Add the Backbone and metric values
            if idx > 0:
                latex_table += r'    '  # Indent for alignment
            latex_table += r' & %s & %s & %s & %s & %.1f & %s \\' % (
                model,
                metric_values[0],
                metric_values[1],
                metric_values[2],
                sigma_value if isinstance(sigma_value, float) else sigma_value,
                minCL_value
            ) + '\n'
        latex_table += r'    \midrule' + '\n'

    # Close the LaTeX table
    latex_table += r'''    \bottomrule
    \end{tabular}
    \vspace{-0.1in}
    \caption{Comparison of EndoFM and ResNet101 across datasets.}
    \label{tab:performance_comparison}
\end{table}
'''

    return latex_table

def save_latex_table(csv_file, metrics, output_tex_file):
    """
    Saves only the LaTeX table to a .tex file, ready for inclusion in an existing LaTeX document.

    :param csv_file: Path to the CSV file containing the data.
    :param metrics: A list of performance metrics to include in the table.
    :param output_tex_file: Path to save the generated .tex file.
    """
    # Generate the LaTeX table
    latex_table = generate_latex_table(csv_file, metrics)

    # Write the LaTeX table to the output file
    with open(output_tex_file, 'w') as tex_file:
        tex_file.write(latex_table)

    print(f"LaTeX table generated and saved to {output_tex_file}")



experiments_root = './Experiments'  # Path to the 'Experiments' directory
metrics_to_include = ['accuracy_with_anomalies', 'nmi_with_anomalies', 'ari_with_anomalies']  # List of metrics to include
best_metrics = ['accuracy_with_anomalies', 'nmi_with_anomalies', 'ari_with_anomalies']  # Metrics to use for selecting best performances


if __name__ == "__main__":

    for best_metric in best_metrics:
        print(f"\nProcessing best metric: {best_metric}")

        # Collect best performances based on the current best_metric
        try:
            best_results_df = collect_best_performances(experiments_root, metrics=metrics_to_include, best_metric=best_metric)
        except Exception as e:
            print(f"Error collecting best performances for {best_metric}: {e}")
            continue

        # Define output CSV and LaTeX filenames
        safe_metric = best_metric.replace('/', '_').replace('\\', '_')
        output_csv_path = f'./best_performances_{safe_metric}.csv'
        output_tex_file = f'./performance_comparison_table_{safe_metric}.tex'

        # Save the best performances to a CSV file
        try:
            best_results_df.to_csv(output_csv_path, index=False)
            print(f"The best performances have been saved to {output_csv_path}")
        except Exception as e:
            print(f"Error saving CSV for {best_metric}: {e}")
            continue

        # Generate and save the LaTeX table
        try:
            save_latex_table(csv_file=output_csv_path, metrics=metrics_to_include, output_tex_file=output_tex_file)
        except Exception as e:
            print(f"Error generating LaTeX table for {best_metric}: {e}")
            continue
    print("\nAll tables have been processed.")