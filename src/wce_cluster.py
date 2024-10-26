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


# def copy_images_to_clusters(minCl, minSpl, sigma, labels, output_dir, dataset_samples, model_name, name, fps):
#     """
#     Copy images to directories based on cluster labels.
#
#     Parameters:
#     - minCl: Minimum cluster size used in HDBSCAN.
#     - minSpl: Minimum samples used in HDBSCAN.
#     - sigma: Sigma value used in Gaussian smoothing.
#     - labels: Cluster labels obtained from HDBSCAN.
#     - output_dir: Base output directory.
#     - dataset_samples: List of dataset samples (image paths and labels).
#     - model_name: Name of the model used for feature extraction.
#     - name: Name of the dataset or experiment.
#     - fps: Frames per second value.
#     """
#     if not any([minCl, minSpl, sigma]):
#         return
#
#     out_dir = os.path.join(output_dir, name, f"{fps}FPS", f"{model_name}_sigma{sigma}_{minCl}_{minSpl}")
#     # Create the output directory if it doesn't exist
#     os.makedirs(out_dir, exist_ok=True)
#
#     # Iterate through each image and copy it to the corresponding cluster directory
#     for i, label in enumerate(labels):
#         cluster_dir = os.path.join(out_dir, f"cluster_{label}" if label != -1 else "noise")
#
#         if not os.path.exists(cluster_dir):
#             os.makedirs(cluster_dir)
#
#         # Get the original image path
#         image_path, _ = dataset_samples[i]
#
#         # Copy the image to the corresponding cluster directory
#         try:
#             shutil.copy(image_path, cluster_dir)
#         except Exception as e:
#             print(f"Error copying {image_path} to {cluster_dir}: {e}")

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
        backbone=Model.ENDO_FM,
        evaluate: bool = False,
        smooth=True,
        save_clusters: bool = False,
        output_dir="./dumps/clustered_images",
        plot_time_series=False,
        student=True,
        draw_plots=True,
        sigmas: list[float] = [6],
        fps=None,
        save_full_fps=False,
        recompute = False
    ):
        if fps is None:
            raise RuntimeError("Please specify FPS")
        self.recompute = recompute
        self.sigmas = sigmas
        self.fps = fps
        self.save_full_fps = save_full_fps  # Whether to save and visualize the full FPS version
        self.plot_time_series = plot_time_series
        self.minCl_values = minCl
        self.minSpl_values = minSpl
        self.evaluate = evaluate
        self.smooth = smooth
        self.save_clusters = save_clusters
        self.name = os.path.basename(dataset_path.rstrip('/'))
        self.draw_plots = draw_plots
        self.model_name = backbone.name
        self.backbone = FeatureGenerator(model_name=backbone, student=student)  # Adjust accordingly

        if backbone in (Model.ENDO_FM,Model.CENDO_FM):
            normalize = v2.Normalize([0.5929, 0.3667, 0.1843], [0.1932, 0.1411, 0.0940])
        elif backbone in (Model.DEPTH_ANY_SMALL, Model.DEPTH_ANY_LARGE, Model.DEPTH_ANY_BASE):
            normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,
                                                             0.225])  # Example normalization for a standard ImageNet-trained model
        elif backbone in (Model.RES_NET_50, Model.RES_NET_101):
            normalize = v2.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Typical normalization used for ResNet
        else:
            raise ValueError(f"Unsupported backbone model: {backbone}")

        preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,  # Apply the normalization specific to the backbone
            v2.Resize(img_size)
        ])
        self.dataset = datasets.ImageFolder(dataset_path, transform=preprocess)
        self.output_dir = output_dir

        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()

        self.data_loader = DataLoader(self.dataset, batch_size=batch_size)

    def estimate_noise_std(self, features):
        # Compute differences between adjacent features
        diffs = np.diff(features, axis=0)
        # Flatten the differences if features are multi-dimensional
        diffs_flat = diffs.flatten()
        # Use Median Absolute Deviation (MAD) for robustness
        mad = np.median(np.abs(diffs_flat - np.median(diffs_flat)))
        # Convert MAD to standard deviation (assuming Gaussian noise)
        noise_std = mad / 0.6745
        return noise_std

    def _extract_features(self):
        """
        Extract features from the dataset or load them if they already exist.
        """
        feature_dir = os.path.join('./dumps/Features', self.name)
        os.makedirs(feature_dir, exist_ok=True)
        feature_file = os.path.join(feature_dir, f'{self.model_name}_{self.fps}FPS_features.npy')
        label_file = os.path.join(feature_dir, f'{self.model_name}_{self.fps}FPS_labels.npy')

        if os.path.exists(feature_file) and os.path.exists(label_file) and not self.recompute:
            print(f"Loading features from {feature_file}")
            self.features = np.load(feature_file)
            self.control_labels = np.load(label_file)
        else:
            extracted_features = []
            control_labels = []
            print("Beginning feature extraction...")

            # First pass: Collect all features and labels
            for img_batch, label_batch in tqdm(self.data_loader, desc="Passing images through backbone", unit="batch"):
                # Generate features using the backbone model
                features = self.backbone.generate(img_batch)
                features = features.view(features.size(0), -1)
                features_np = features.cpu().numpy()

                # Append features and labels to the list without smoothing
                extracted_features.append(features_np)
                control_labels.append(label_batch.cpu().numpy())

            # Stack the transformed data and control labels into final arrays
            self.features = np.vstack(extracted_features)
            self.control_labels = np.hstack(control_labels)

            # Save features and labels for future use
            np.save(feature_file, self.features)
            np.save(label_file, self.control_labels)
            print(f"Features saved to {feature_file}")

    def _visualize_original_distribution(self, sigma):
        """
        Visualize the original distribution with raw and smoothed features using original labels.
        This is saved only once per model and FPS.
        """
        fps_folder = f"{self.fps}FPS"
        output_dir_base = os.path.join("./dumps/Plots", self.name, fps_folder, self.model_name, "FullFPSClusters")
        os.makedirs(output_dir_base, exist_ok=True)

        # Check if the plots already exist
        output_path_raw = os.path.join(output_dir_base, f"tsne_original_labels_raw_{self.name}.svg")
        output_path_smoothed = os.path.join(output_dir_base, f"tsne_original_labels_smoothed_{self.name}_sigma{sigma}.svg")
        if os.path.exists(output_path_raw) and os.path.exists(output_path_smoothed) and not self.recompute:
            print(f"Original distribution plots already exist: {output_path_raw}")
            return

        print(f"Visualizing original distribution with sigma={sigma}")

        # Time steps
        time_steps = np.arange(len(self.control_labels))

        # Visualize raw features with original labels
        if not os.path.exists(output_path_raw):
            visualize_data(
                data=self.raw_features,
                labels=self.control_labels,
                time_steps=time_steps,
                title=f"{self.name}",
                output_dir=output_dir_base,
                filename_suffix=f"original_labels_raw_{self.name}",
                control_labels=self.control_labels,
                classes=self.dataset.classes
            )

        # Visualize smoothed features with original labels
        visualize_data(
            data=self.transformed_data,
            labels=self.control_labels,
            time_steps=time_steps,
            title=f"{self.name} Smoothed (sigma={sigma})",
            output_dir=output_dir_base,
            filename_suffix=f"original_labels_smoothed_{self.name}_sigma{sigma}",
            control_labels=self.control_labels,
            classes=self.dataset.classes
        )

    def _process_sigma(self, sigma):
        print(f"Processing sigma: {sigma}")

        # Apply Gaussian filter to features
        radius = int(3 * sigma)
        transformed_data = gaussian_filter(self.features, sigma=sigma, radius=radius)
        self.transformed_data = transformed_data  # Assign transformed data

        # Visualize the original distribution with original labels (only once per sigma)
        if self.draw_plots:
            self._visualize_original_distribution(sigma)

        # Prepare combinations of minCl and minSpl
        combinations = list(product(self.minCl_values, self.minSpl_values))

        # Store transformed_data and control_labels in a memmap to share between processes
        temp_folder = tempfile.mkdtemp()
        data_filename_memmap = os.path.join(temp_folder, 'data_memmap')
        labels_filename_memmap = os.path.join(temp_folder, 'labels_memmap')

        joblib.dump(transformed_data, data_filename_memmap)
        memmap_data = joblib.load(data_filename_memmap, mmap_mode='r')

        joblib.dump(self.control_labels, labels_filename_memmap)
        memmap_labels = joblib.load(labels_filename_memmap, mmap_mode='r')

        # Prepare dataset samples for image copying
        dataset_samples = self.dataset.samples

        # Pass necessary parameters explicitly to avoid referencing self
        process_combination_params = {
            'sigma': sigma,
            'memmap_data': memmap_data,
            'memmap_labels': memmap_labels,
            'name': self.name,
            'model_name': self.model_name,
            'fps': self.fps,
            'evaluate': self.evaluate,
            'save_full_fps': self.save_full_fps,
            'draw_plots': self.draw_plots,
            'save_clusters': self.save_clusters,
            'output_dir': self.output_dir,
            'classes': self.dataset.classes,
            'transformed_data': transformed_data,  # For visualization
            'raw_features': self.raw_features,     # For visualization
            'dataset_samples': dataset_samples     # For copying images
        }

        # Parallelize over combinations
        results = Parallel(n_jobs=4)(
            delayed(process_combination)(minCl, minSpl, process_combination_params)
            for minCl, minSpl in combinations
        )

        # Filter out None values and collect evaluation results
        evaluation_results = [metrics for metrics in results if metrics is not None]

        # After all computations, create a DataFrame and save or display it
        if self.evaluate and evaluation_results:
            results_df = pd.DataFrame(evaluation_results)
            print("\nEvaluation Results:")
            print(results_df)

            # Save the results to a CSV file
            eval_dir = os.path.join("./dumps/Evaluation", self.name, self.model_name, f"Sigma{sigma}")
            os.makedirs(eval_dir, exist_ok=True)
            output_path = os.path.join(eval_dir, f"evaluation_results_{self.name}_sigma{sigma}.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Evaluation results saved to {output_path}")

    def apply(self):
        self._extract_features()

        # Store raw features and original labels before any modifications
        self.raw_features = self.features.copy()

        # Unload the model to free GPU memory
        del self.backbone.model
        torch.cuda.empty_cache()

        # Parallelize over sigmas
        Parallel(n_jobs=4)(
            delayed(self._process_sigma)(sigma) for sigma in self.sigmas
        )