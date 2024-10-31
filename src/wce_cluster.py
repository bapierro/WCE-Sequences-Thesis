import os
import shutil
import tempfile

import joblib
from joblib import Memory
import torch
import torchvision.transforms as transforms
import pandas as pd
from mpmath.libmp import normalize
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
import hdbscan
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from joblib import Parallel, delayed

from model_name import Model
from feature_generator import FeatureGenerator


def evaluate_clustering(true_labels, predicted_labels):
    """
    Evaluate clustering performance metrics.

    Parameters:
    - true_labels: Ground truth labels.
    - predicted_labels: Labels predicted by the clustering algorithm.

    Returns:
    - A dictionary containing accuracy, NMI, and ARI with and without anomalies.
    """
    
    # Calculate accuracy with anomalies
    label_mapping = np.zeros_like(predicted_labels)
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)

    # Create a cost matrix for the Hungarian algorithm
    cost_matrix = np.zeros((len(unique_true), len(unique_pred)))
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            cost_matrix[i, j] = np.sum((true_labels == true_label) & (predicted_labels == pred_label))

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    for i, j in zip(row_ind, col_ind):
        label_mapping[predicted_labels == unique_pred[j]] = unique_true[i]

    accuracy_with_anomalies = accuracy_score(true_labels, label_mapping)
    nmi_with_anomalies = normalized_mutual_info_score(true_labels, predicted_labels)
    ari_with_anomalies = adjusted_rand_score(true_labels, predicted_labels)

    mask = predicted_labels != -1
    if np.any(mask):
        accuracy_without_anomalies = accuracy_score(true_labels[mask], predicted_labels[mask])
        nmi_without_anomalies = normalized_mutual_info_score(true_labels[mask], predicted_labels[mask])
        ari_without_anomalies = adjusted_rand_score(true_labels[mask], predicted_labels[mask])
    else:
        # If all labels are -1, set metrics to None or 0
        accuracy_without_anomalies = None
        nmi_without_anomalies = None
        ari_without_anomalies = None

    # Print statements can be kept or removed based on preference
    print("Evaluation with anomalies (-1 included):")
    print(f"Accuracy: {accuracy_with_anomalies:.4f}")
    print(f"NMI: {nmi_with_anomalies:.4f}")
    print(f"ARI: {ari_with_anomalies:.4f}\n")

    if accuracy_without_anomalies is not None:
        print("Evaluation without anomalies (-1 excluded):")
        print(f"Accuracy: {accuracy_without_anomalies:.4f}")
        print(f"NMI: {nmi_without_anomalies:.4f}")
        print(f"ARI: {ari_without_anomalies:.4f}")
    else:
        print("All points are labeled as anomalies (-1). Cannot compute metrics without anomalies.")

    # Return the metrics in a dictionary
    return {
        'accuracy_with_anomalies': accuracy_with_anomalies,
        'nmi_with_anomalies': nmi_with_anomalies,
        'ari_with_anomalies': ari_with_anomalies,
        'accuracy_without_anomalies': accuracy_without_anomalies,
        'nmi_without_anomalies': nmi_without_anomalies,
        'ari_without_anomalies': ari_without_anomalies
    }


def visualize_data(data, labels, time_steps, title, output_dir, filename_suffix, minCl=None, minSpl=None,
                  sigma=None, control_labels=None, classes=None):
    """
    General function to visualize data using t-SNE and create 2D and 3D plots.

    Parameters:
    - data: The feature data to visualize.
    - labels: Cluster labels for the data.
    - time_steps: Time steps or frame indices corresponding to the data.
    - title: Title for the plots.
    - output_dir: Directory to save plots.
    - filename_suffix: Suffix to add to output filenames.
    - minCl, minSpl, sigma: Clustering parameters for titles.
    - control_labels: Optional control labels for plotting.
    - classes: List of class names in the dataset (e.g., ['class0', 'class1', ...]).
    """

    # Prepare TSNE data
    tsne = TSNE(n_components=2, random_state=20)
    tsne_data = tsne.fit_transform(data)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'time': time_steps,
        'tsne_1': tsne_data[:, 0],
        'tsne_2': tsne_data[:, 1],
        'cluster': labels
    })

    # Generate color map
    unique_labels = np.unique(labels)
    cluster_labels = [label for label in unique_labels if label != -1]
    colors = px.colors.qualitative.G10
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(cluster_labels)}
    color_map[-1] = 'black'

    # Map labels to colors
    plot_data['color'] = plot_data['cluster'].map(color_map)

    # ----- 2D TSNE Plot -----
    plt.figure(figsize=(12, 10))
    if control_labels is not None and classes is not None:
        # Assuming control_labels are numerical and correspond to classes
        sns.scatterplot(
            x=plot_data['tsne_1'],
            y=plot_data['tsne_2'],
            hue=[classes[label] for label in control_labels],
            palette=sns.color_palette("colorblind", n_colors=len(classes)),
            legend="full",
            alpha=0.6
        )
    else:
        sns.scatterplot(
            x=plot_data['tsne_1'],
            y=plot_data['tsne_2'],
            hue=plot_data['cluster'],
            palette=color_map,
            legend="full",
            alpha=0.6
        )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0., fontsize="small")
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # Save the 2D plot
    output_path_2d = os.path.join(output_dir, f"tsne_{filename_suffix}.svg")
    plt.savefig(output_path_2d, format='svg', bbox_inches='tight')
    print(f"TSNE plot saved to {output_path_2d}")
    plt.close()

    # ----- 3D Plotly Plot -----
    fig = go.Figure()

    # Loop over each cluster to create a separate trace
    for cluster_label in plot_data['cluster'].unique():
        cluster_data = plot_data[plot_data['cluster'] == cluster_label]
        fig.add_trace(go.Scatter3d(
            y=cluster_data['time'],
            x=cluster_data['tsne_1'],
            z=cluster_data['tsne_2'],
            mode='markers',
            marker=dict(
                size=3,
                color=color_map[cluster_label],
                opacity=0.6
            ),
            name=f'Cluster {cluster_label}',
            text=[f"Frame: {t}, Cluster: {c}" for t, c in zip(cluster_data['time'], cluster_data['cluster'])]
        ))

    fig.update_layout(
        scene=dict(
            yaxis_title='Frame',
            xaxis_title='t-SNE 1',
            zaxis_title='t-SNE 2',
        ),
        title=title,
        width=800,
        height=800,
        legend_title='Cluster Labels'
    )

    # Save the interactive 3D figure
    output_path_3d = os.path.join(output_dir, f"interactive_3d_plot_{filename_suffix}.html")
    fig.write_html(output_path_3d)
    print(f"Interactive 3D plot saved to {output_path_3d}")




def visualize_full_fps(minCl, minSpl, sigma, labels, transformed_data, raw_features, name, fps, model_name,
                      classes, output_dir):
    """
    Visualize the full FPS data with clusters.
    This function generates plots for both transformed and raw features.

    Parameters:
    - minCl: Minimum cluster size used in HDBSCAN.
    - minSpl: Minimum samples used in HDBSCAN.
    - sigma: Sigma value used in Gaussian smoothing.
    - labels: Cluster labels obtained from HDBSCAN.
    - transformed_data: The smoothed feature data after applying Gaussian filter.
    - raw_features: The original feature data before smoothing.
    - name: Name of the dataset or experiment.
    - fps: Frames per second value.
    - model_name: Name of the model used for feature extraction.
    - classes: List of class names in the dataset (e.g., ['class0', 'class1', ...]).
    - output_dir: Base output directory to save plots.
    """

    fps_folder = f"{fps}FPS"
    output_dir_base = os.path.join(output_dir, name, fps_folder, model_name, "FullFPSClusters")
    os.makedirs(output_dir_base, exist_ok=True)

    # Time steps
    time_steps = np.arange(len(labels))

    # Visualize full FPS clusters on transformed features
    visualize_data(
        data=transformed_data,
        labels=labels,
        time_steps=time_steps,
        title=f"{name} Smoothed (minCl={minCl}, minSpl={minSpl}, sigma={sigma})",
        output_dir=output_dir_base,
        filename_suffix=f"full_fps_clusters_transformed_{name}_sigma{sigma}_minCl{minCl}_minSpl{minSpl}",
        minCl=minCl,
        minSpl=minSpl,
        sigma=sigma,
        control_labels=None,
        classes=classes  # Pass classes for consistency, even if control_labels is None
    )

    # Visualize full FPS clusters on raw features
    visualize_data(
        data=raw_features,
        labels=labels,
        time_steps=time_steps,
        title=f"Clusters {name} (minCl={minCl}, minSpl={minSpl}, sigma={sigma})",
        output_dir=output_dir_base,
        filename_suffix=f"full_fps_clusters_raw_{name}_sigma{sigma}_minCl{minCl}_minSpl{minSpl}",
        minCl=minCl,
        minSpl=minSpl,
        sigma=sigma,
        control_labels=None,
        classes=classes  # Pass classes for consistency
    )


def process_combination(minCl, minSpl, params):
    """
    Process a single combination of minCl and minSpl.

    Parameters:
    - minCl: Minimum cluster size for HDBSCAN.
    - minSpl: Minimum samples for HDBSCAN.
    - params: Dictionary containing all necessary parameters.

    Returns:
    - metrics: Dictionary of evaluation metrics, or None if evaluation is not required.
    """

    sigma = params['sigma']
    data = params['memmap_data']
    control_labels = params['memmap_labels']
    name = params['name']
    model_name = params['model_name']
    fps = params['fps']
    evaluate = params['evaluate']
    save_full_fps = params['save_full_fps']
    draw_plots = params['draw_plots']
    save_clusters = params['save_clusters']
    output_dir = params['output_dir']
    classes = params['classes']
    transformed_data = params['transformed_data']
    raw_features = params['raw_features']
    dataset_samples = params['dataset_samples']  # Needed for copying images

    print(f"Processing minCl: {minCl}, minSpl: {minSpl} for sigma: {sigma}")

    # Create a new instance of HDBSCAN for each combination
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=minCl,
        min_samples=minSpl,
        memory=Memory(location="./dumps/cache", verbose=0)
    )
    hdbscan_clusterer.fit(data)
    labels = hdbscan_clusterer.labels_

    # Save control labels once per sigma (in a thread-safe manner)
    labels_dir_full = os.path.join('./dumps/Labels', name, model_name, f"{fps}FPS")
    os.makedirs(labels_dir_full, exist_ok=True)
    full_control_labels_path = os.path.join(labels_dir_full, f"full_control_labels_{name}.npy")
    try:
        # Use file lock or check existence to prevent race conditions
        if not os.path.exists(full_control_labels_path):
            np.save(full_control_labels_path, control_labels)
            print(f"Control labels saved to {full_control_labels_path}")
    except Exception as e:
        print(f"Error saving control labels: {e}")

    metrics = None
    if evaluate:
        # Evaluate clustering
        metrics = evaluate_clustering(control_labels, labels)
        metrics['sigma'] = sigma
        metrics['minCl'] = minCl
        metrics['minSpl'] = minSpl

    if save_full_fps and draw_plots:
        # Visualize clusters
        visualize_full_fps(
            minCl=minCl,
            minSpl=minSpl,
            sigma=sigma,
            labels=labels,
            transformed_data=transformed_data,
            raw_features=raw_features,
            name=name,
            fps=fps,
            model_name=model_name,
            classes=classes,
            output_dir=output_dir
        )

    # if save_clusters:
    #     # Copy images to clusters
    #     copy_images_to_clusters(
    #         minCl=minCl,
    #         minSpl=minSpl,
    #         sigma=sigma,
    #         labels=labels,
    #         output_dir=output_dir,
    #         dataset_samples=dataset_samples,
    #         model_name=model_name,
    #         name=name,
    #         fps=fps
    #     )

    return metrics  # Return the metrics for this combination


class WCECluster:
    def __init__(
            self,
            dataset_path: str,
            minCl: list[int] = [30],
            minSpl: list[int] = [1],
            batch_size: int = 32,
            img_size: int = 224,
            backbones=None,  # Accept a list of models
            evaluate: bool = False,
            smooth=True,
            student=True,
            draw_plots=True,
            sigmas: list[float] = [6],
            fps=None,
            recompute=False
    ):
        """
        Initialize the WCECluster class.

        Parameters:
        - dataset_path: Path to the dataset directory.
        - minCl: List of minimum cluster sizes for HDBSCAN.
        - minSpl: List of minimum samples for HDBSCAN.
        - batch_size: Batch size for DataLoader.
        - img_size: Image size for transformations.
        - backbones: List of backbone models for feature extraction.
        - evaluate: Whether to compute evaluation metrics.
        - smooth: Whether to apply Gaussian smoothing.
        - student: Whether to use student model features.
        - draw_plots: Whether to generate and save plots.
        - sigmas: List of sigma values for Gaussian smoothing.
        - fps: Frames per second value.
        - recompute: Whether to recompute features if they exist.
        """
        if fps is None:
            raise RuntimeError("Please specify FPS")
        if backbones is None or not isinstance(backbones, list):
            raise ValueError("Please provide a list of backbone models.")

        self.recompute = recompute
        self.sigmas = sigmas
        self.fps = fps
        self.minCl_values = minCl
        self.minSpl_values = minSpl
        self.evaluate = evaluate
        self.smooth = smooth
        self.backbones = backbones  # Now a list of models
        self.student = student
        self.batch_size = batch_size
        self.img_size = img_size
        self.name = os.path.basename(dataset_path.rstrip('/'))
        self.draw_plots = draw_plots

        # Initialize DataLoader
        self.dataset = datasets.ImageFolder(dataset_path)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        # Set up Seaborn aesthetics
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()

    def _get_preprocess(self, backbone):
        """
        Get the preprocessing transformation specific to the backbone model.
        """
        if backbone in (Model.ENDO_FM, Model.CENDO_FM):
            normalize = v2.Normalize([0.5929, 0.3667, 0.1843], [0.1932, 0.1411, 0.0940])
        elif backbone in (Model.DEPTH_ANY_SMALL, Model.DEPTH_ANY_LARGE, Model.DEPTH_ANY_BASE):
            normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif backbone in (Model.RES_NET_50, Model.RES_NET_101, Model.RES_NET_18, Model.RES_NET_34, Model.RES_NET_152):
            normalize = v2.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}")

        # Define data transformations
        preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,  # Apply the normalization specific to the backbone
            v2.Resize(self.img_size)
        ])
        return preprocess

    def _extract_features(self, backbone):
        """
        Extract features using the specified backbone model.

        Parameters:
        - backbone: The backbone model to use for feature extraction.

        Returns:
        - features: Extracted features as a NumPy array.
        """
        feature_dir = os.path.join('./dumps/Features', self.name)
        os.makedirs(feature_dir, exist_ok=True)
        feature_file = os.path.join(feature_dir, f'{backbone.name}_{self.fps}FPS_features.npy')

        if os.path.exists(feature_file) and not self.recompute:
            print(f"Loading features from {feature_file} for model {backbone.name}")
            features = np.load(feature_file)
        else:
            # Initialize the feature generator
            feature_generator = FeatureGenerator(model_name=backbone, student=self.student).to(DEVICE)
            feature_generator.model.eval()  # Ensure the model is in evaluation mode

            # Update the DataLoader with the appropriate transforms
            preprocess = self._get_preprocess(backbone)
            self.dataset.transform = preprocess
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

            extracted_features = []
            print(f"Beginning feature extraction with {backbone.name}...")

            # Collect all features
            for img_batch, _ in tqdm(data_loader, desc=f"Extracting features with {backbone.name}", unit="batch"):
                # Generate features using the backbone model
                features_batch = feature_generator.generate(img_batch)
                features_batch = features_batch.view(features_batch.size(0), -1)
                features_np = features_batch.cpu().numpy()

                # Append features to the list
                extracted_features.append(features_np)

            # Stack the transformed data into final array
            features = np.vstack(extracted_features)

            # Save features for future use
            np.save(feature_file, features)
            print(f"Features saved to {feature_file} for model {backbone.name}")

            # Clean up to free memory
            del feature_generator
            torch.cuda.empty_cache()

        return features

    def _process_sigma(self, sigma, features, backbone_name):
        """
        Process the data for a specific sigma value, including smoothing, clustering, evaluation, and visualization.

        Parameters:
        - sigma: Sigma value for Gaussian smoothing.
        - features: Feature array to process.
        - backbone_name: Name of the backbone model used.
        """
        print(f"Processing sigma: {sigma} for model: {backbone_name}")

        # Apply Gaussian filter to features
        if self.smooth:
            transformed_data = gaussian_filter(features, sigma=sigma)
        else:
            transformed_data = features.copy()
        self.transformed_data = transformed_data  # Assign transformed data

        # Apply PCA for dimensionality reduction if needed
        # Uncomment the following lines if you want to apply PCA
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=100)
        # transformed_data = pca.fit_transform(transformed_data)
        # print(f"PCA reduced data shape to {transformed_data.shape}")

        # Prepare combinations of minCl and minSpl
        combinations = list(product(self.minCl_values, self.minSpl_values))

        # Store transformed_data in a memmap to share between processes
        temp_folder = tempfile.mkdtemp()
        data_filename_memmap = os.path.join(temp_folder, 'data_memmap')
        joblib.dump(transformed_data, data_filename_memmap)
        memmap_data = joblib.load(data_filename_memmap, mmap_mode='r')

        # Pass necessary parameters explicitly to avoid referencing self
        process_combination_params = {
            'sigma': sigma,
            'memmap_data': memmap_data,
            'name': self.name,
            'model_name': backbone_name,
            'fps': self.fps,
            'evaluate': self.evaluate,
            'draw_plots': self.draw_plots,
            'output_dir': self.output_dir,
            'transformed_data': transformed_data,  # For visualization
            'raw_features': features,  # For potential future use
            'dataset_samples': self.dataset.samples  # List of (image_path, class_idx)
        }

        # Parallelize over combinations
        results = Parallel(n_jobs=4)(
            delayed(process_combination)(minCl, minSpl, process_combination_params)
            for minCl, minSpl in combinations
        )

        # Collect evaluation results
        evaluation_results = [metrics for metrics in results if metrics]

        # After all computations, create a DataFrame and save or display it
        if self.evaluate and evaluation_results:
            # Separate HDBSCAN and Random_Segmentation metrics
            hdbscan_metrics = []
            random_metrics = []
            for metrics in evaluation_results:
                if 'HDBSCAN' in metrics:
                    hdbscan_metrics.append(metrics['HDBSCAN'])
                if 'Random_Segmentation' in metrics:
                    random_metrics.append(metrics['Random_Segmentation'])

            results_hdbscan_df = pd.DataFrame(hdbscan_metrics)
            results_random_df = pd.DataFrame(random_metrics)
            print("\nEvaluation Results for HDBSCAN:")
            print(results_hdbscan_df)
            print("\nEvaluation Results for Random Segmentation:")
            print(results_random_df)

            # Merge the two DataFrames for comparison
            results_hdbscan_df['method'] = 'HDBSCAN'
            results_random_df['method'] = 'Random_Segmentation'
            combined_results_df = pd.concat([results_hdbscan_df, results_random_df], ignore_index=True)

            # Save the combined results to a CSV file
            eval_dir = os.path.join("./dumps/Evaluation", self.name, backbone_name, f"Sigma{sigma}")
            os.makedirs(eval_dir, exist_ok=True)
            output_path = os.path.join(eval_dir, f"evaluation_results_combined_sigma{sigma}.csv")
            combined_results_df.to_csv(output_path, index=False)
            print(f"Combined evaluation results saved to {output_path}")

        # Clean up temporary files
        try:
            shutil.rmtree(temp_folder)
        except Exception as e:
            print(f"Could not delete temp folder {temp_folder}: {e}")

    def apply(self):
        """
        Execute the clustering pipeline: feature extraction, clustering, evaluation, and visualization for each backbone.
        """
        for backbone in self.backbones:
            print(f"\nProcessing with model: {backbone.name}")
            self.model_name = backbone.name  # Update model_name for output paths

            # Extract features for the current model
            features = self._extract_features(backbone)

            # Store raw features before any modifications
            self.raw_features = features.copy()

            # Update output directory for the current model
            fps_folder = f"{self.fps}FPS"
            self.output_dir = os.path.join("./dumps/Plots", self.name, fps_folder, self.model_name, "FullFPSClusters")
            os.makedirs(self.output_dir, exist_ok=True)

            # Parallelize over sigmas
            Parallel(n_jobs=4)(
                delayed(self._process_sigma)(sigma, features, self.model_name) for sigma in self.sigmas
            )