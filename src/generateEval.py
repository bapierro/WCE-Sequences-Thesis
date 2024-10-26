from wce_cluster import WCECluster
from model_name import Model
import pandas as pd
import os

base_path = "../evaluation_data/kvasir_capsule_sequences/selection/selection"
folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
minCLs = [50,100,300,500,1000]

def merge_model_evaluation_csvs(base_path, minCL):
    """
    Merge all CSV files within each model's Sigma directory across all datasets,
    aggregating per sigma and keeping models separate. The sigma directories are
    placed inside a directory called 'ExperimentsMinCl{minCL}'.

    :param base_path: Path to the Evaluation directory containing dataset directories.
    :param minCL: The value of minCL used in directory naming.
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
        combined_data[key] = pd.concat(dfs, ignore_index=True)

    return combined_data

if __name__ == "__main__":
    for folder in folders:
        print("-----------------------")
        print(f"Section: {folder.split(os.sep)[-1]}")

        # WCECluster(folder, minCl=minCLs, batch_size=64, smooth=True, fps=30,
        #            backbone=Model.DEPTH_ANY_BASE, save_full_fps=True, evaluate=True, draw_plots=False,
        #            sigmas=[0, 1, 2, 3, 4, 5]).apply()


        WCECluster(folder, minCl=minCLs, batch_size=64, smooth=True, fps=30,
                   backbone=Model.CENDO_FM, save_full_fps=True, evaluate=True, draw_plots=False,
                   sigmas=[0, 1, 2, 3, 4, 5],recompute=False).apply()



    for minCl in minCLs:
        # Example usage
        base_evaluation_path = "./dumps/Evaluation/"
        merged_results = merge_model_evaluation_csvs(base_evaluation_path, minCl)

        # Create the 'ExperimentsMinCl{minCL}' directory
        experiments_dir = f"./ExperimentsMinCl{minCl}"
        os.makedirs(experiments_dir, exist_ok=True)

        # Save the combined results into separate files for each model and sigma
        for (model, sigma), combined_df in merged_results.items():
            # Clean the sigma name to remove any file system incompatible characters if necessary
            safe_sigma = sigma.replace('/', '_').replace('\\', '_')
            # Create directory for this Sigma within 'ExperimentsMinCl{minCL}' if it doesn't exist
            sigma_dir = os.path.join(experiments_dir, safe_sigma)
            os.makedirs(sigma_dir, exist_ok=True)
            # Build the output path
            output_path = os.path.join(sigma_dir, f"combined_{model}_{safe_sigma}_results.csv")
            combined_df.to_csv(output_path, index=False)

        print(f"CSV files have been merged and saved into directories per Sigma within 'ExperimentsMinCl{minCl}'.")