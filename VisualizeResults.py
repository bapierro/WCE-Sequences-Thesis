import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Directory containing the CSV summaries
    summaries_dir = './Summaries/Median'

    # Define Sigma values corresponding to the filenames
    sigma_mapping = {
        'Sigma0': 0, 'Sigma1': 1, 'Sigma5': 5, 'Sigma10': 10,
        'Sigma25': 25, 'Sigma50': 50, 'Sigma100': 100, 'Sigma200': 200
    }

    # Define a mapping for metric renaming
    metric_renaming = {
        'accuracy': 'Accuracy',
        'nmi': 'NMI',
        'ari': 'ARI',
        'Davies-Bouldin Index': 'DBI',
        'Calinski–Harabasz Index': 'CHI',
        'Density Based Clustering Validation Index': 'DBCVI',
        'Noise Percentage': 'Noise %',
        # 'Temporal Purity': 'Temporal Purity'
    }

    # List of metrics to keep (exclude Intercluster and Intracluster distances)
    metrics_to_keep = list(metric_renaming.keys())

    # Initialize an empty DataFrame to store all data
    all_sigma_data = pd.DataFrame()

    # Iterate over files in the summaries directory and append data
    for filename in os.listdir(summaries_dir):
        if filename.endswith('.csv'):
            sigma_key = next((key for key in sigma_mapping if key in filename), None)
            if sigma_key:
                sigma_value = sigma_mapping[sigma_key]
                file_path = os.path.join(summaries_dir, filename)
                
                # Read the CSV file
                try:
                    sigma_data = pd.read_csv(file_path, index_col=0)  # Metric names are in the index
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue  # Skip to the next file
                
                # Filter out unwanted metrics
                available_metrics = sigma_data.index.intersection(metrics_to_keep)
                missing_metrics = set(metrics_to_keep) - set(available_metrics)
                if missing_metrics:
                    print(f"Warning: Missing metrics in file {filename}: {missing_metrics}")
                sigma_data = sigma_data.loc[available_metrics]
                
                # Rename metrics
                sigma_data.rename(index=metric_renaming, inplace=True)
                
                # Reshape the data from wide to long format
                sigma_data = sigma_data.reset_index().melt(id_vars='index', var_name='Model', value_name='Value')
                
                # Add Sigma column
                sigma_data['Sigma'] = sigma_value  # Add Sigma column
                
                # Rename 'index' to 'Metric'
                sigma_data.rename(columns={'index': 'Metric'}, inplace=True)
                
                # Append to the main DataFrame
                all_sigma_data = pd.concat([all_sigma_data, sigma_data], ignore_index=True)

    # Initial Data Inspection
    print("Initial Data Sample:")
    print(all_sigma_data.head())
    print("\nInitial Data Shape:", all_sigma_data.shape)

    # Remove duplicates based on 'Sigma', 'Metric', and 'Model' only
    all_sigma_data = all_sigma_data.drop_duplicates(subset=['Sigma', 'Metric', 'Model'])

    # Ensure 'Value' is numeric
    all_sigma_data['Value'] = pd.to_numeric(all_sigma_data['Value'], errors='coerce')

    # Check for any remaining duplicates
    duplicates = all_sigma_data.duplicated(subset=['Sigma', 'Metric', 'Model'], keep=False)
    if duplicates.any():
        print("Duplicates detected after deduplication. Aggregating by mean.")
        all_sigma_data = all_sigma_data.groupby(['Sigma', 'Metric', 'Model'], as_index=False).mean()

    # Final Data Inspection
    print("\nData Shape After Deduplication and Aggregation:", all_sigma_data.shape)
    print("Final Data Sample:")
    print(all_sigma_data.head())

    # Compute correlation between Sigma and validation metrics for each model
    models = all_sigma_data['Model'].unique()
    correlation_results = {}

    for model in models:
        model_data = all_sigma_data[all_sigma_data['Model'] == model]
        
        # Pivot the data to have 'Sigma' as index and 'Metric' as columns
        pivoted_data = model_data.pivot(index='Sigma', columns='Metric', values='Value').reset_index()
        
        # Drop rows with NaN values to avoid issues in correlation computation
        pivoted_data = pivoted_data.dropna()
        
        # Verify if 'Sigma' is present in pivoted_data
        if 'Sigma' not in pivoted_data.columns:
            print(f"Warning: 'Sigma' column missing for model {model}. Skipping correlation computation.")
            continue
        
        # Compute correlation of 'Sigma' with each metric
        correlation = pivoted_data.corr().loc['Sigma', :]  # Correlation of Sigma with each metric
        
        # Store the correlation results for the current model
        correlation_results[model] = correlation

    # Combine results into a single DataFrame for comparison
    sigma_correlation_summary = pd.DataFrame(correlation_results)

    # Remove 'Sigma' from index if present
    sigma_correlation_summary = sigma_correlation_summary.drop(index='Sigma', errors='ignore')

    output_path = './Sigma_Correlation_Summary.csv'
    sigma_correlation_summary.to_csv(output_path)
    print(f"\nCorrelation summary saved to {output_path}")

    # Display the result
    print("\nCorrelation Summary:")
    print(sigma_correlation_summary)

    # Plotting the correlation matrix
    plt.figure(figsize=(12, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        sigma_correlation_summary,
        annot=True,
        cmap="BrBG",
        linewidths=0,
        center=0,
        annot_kws={'size': 16}
        # cbar_kws={"shrink": .5},
    )

    plt.title('Effects of Sigma on Clustering Quality', fontsize=16)
    plt.xlabel('', fontsize=14)
    plt.ylabel('', fontsize=14)

    plt.tight_layout()
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14)
    # Save the plot
    heatmap_path = './Sigma_Correlation_Heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to {heatmap_path}")
    
    heatmap_path = './Sigma_Correlation_Heatmap.pdf'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to {heatmap_path}")

    # Display the plot
    plt.show()