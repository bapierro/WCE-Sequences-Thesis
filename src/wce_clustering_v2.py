import os
import shutil
import tempfile
from itertools import product

import joblib
from joblib import Memory, Parallel, delayed
import torch
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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
import plotly.express as px

from model_name import Model
from feature_generator import FeatureGenerator
import pandas as pd

# -----------------------------------------
# Evaluation Metric Functions
# -----------------------------------------

def temporal_purity_score(labels):
    """
    Compute the Temporal Purity Score for clustering labels.

    Parameters:
    - labels: Array of cluster labels for each frame.

    Returns:
    - average_purity: Average temporal purity across all clusters.
    """
    unique_clusters = set(labels)
    unique_clusters.discard(-1)  # Exclude noise

    purities = []
    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) == 0:
            continue

        # Identify continuous sequences
        sequences = np.split(cluster_indices, np.where(np.diff(cluster_indices) != 1)[0] + 1)
        # Find the longest continuous sequence
        longest_sequence = max(sequences, key=len)
        purity = len(longest_sequence) / len(cluster_indices)
        purities.append(purity)

    average_purity = np.mean(purities) if purities else 0
    return average_purity


def compute_cluster_distances(features, labels):
    """
    Compute intercluster and intracluster distances.

    Parameters:
    - features: Array of feature vectors.
    - labels: Array of cluster labels.

    Returns:
    - intercluster_distance: Average minimum distance between clusters.
    - intracluster_distance: Average maximum distance within each cluster.
    """
    unique_clusters = set(labels)
    unique_clusters.discard(-1)  # Exclude noise

    inter_distances = []
    intr_distances = []

    # Precompute pairwise distances
    distance_matrix = pairwise_distances(features)

    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) < 2:
            continue  # Skip clusters with a single point

        # Intracluster: Maximum distance within the cluster
        intr_distance = np.max(distance_matrix[np.ix_(cluster_indices, cluster_indices)])
        intr_distances.append(intr_distance)

        # Intercluster: Minimum distance to other clusters
        other_clusters = unique_clusters - {cluster}
        min_dist = np.inf
        for other in other_clusters:
            other_indices = np.where(labels == other)[0]
            if len(other_indices) == 0:
                continue
            current_min = np.min(distance_matrix[np.ix_(cluster_indices, other_indices)])
            if current_min < min_dist:
                min_dist = current_min
        if min_dist < np.inf:
            inter_distances.append(min_dist)

    intercluster_distance = np.mean(inter_distances) if inter_distances else 0
    intracluster_distance = np.mean(intr_distances) if intr_distances else 0

    return intercluster_distance, intracluster_distance


def perform_random_segmentation(num_frames, num_clusters, random_state=None):
    """
    Perform random segmentation by dividing the video into contiguous clusters.

    Parameters:
    - num_frames: Total number of frames.
    - num_clusters: Number of clusters (segments) to create.
    - random_state: Seed for reproducibility.

    Returns:
    - random_labels: Array of cluster labels for each frame.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if num_clusters <= 0:
        raise ValueError("Number of clusters must be positive.")
    if num_clusters > num_frames:
        raise ValueError("Number of clusters cannot exceed number of frames.")

    # If num_clusters is 1, assign all frames to one cluster
    if num_clusters == 1:
        return np.zeros(num_frames, dtype=int)

    # Randomly select (num_clusters - 1) unique boundaries
    boundaries = np.sort(np.random.choice(range(1, num_frames), size=num_clusters - 1, replace=False))

    # Initialize labels
    random_labels = np.zeros(num_frames, dtype=int)

    # Assign cluster labels based on boundaries
    current_cluster = 0
    prev_boundary = 0
    for boundary in boundaries:
        random_labels[prev_boundary:boundary] = current_cluster
        current_cluster += 1
        prev_boundary = boundary
    random_labels[prev_boundary:] = current_cluster  # Assign remaining frames to the last cluster

    return random_labels


def compute_intra_cluster_variance(features, labels):
    """
    Compute the average intra-cluster variance.

    Parameters:
    - features: Array of feature vectors.
    - labels: Array of cluster labels.

    Returns:
    - average_intra_variance: Average variance within clusters.
    """
    unique_clusters = set(labels)
    unique_clusters.discard(-1)  # Exclude noise

    intra_variances = []
    for cluster in unique_clusters:
        cluster_features = features[labels == cluster]
        centroid = cluster_features.mean(axis=0)
        variance = np.mean(np.linalg.norm(cluster_features - centroid, axis=1) ** 2)
        intra_variances.append(variance)

    average_intra_variance = np.mean(intra_variances) if intra_variances else 0
    return average_intra_variance


def compute_davies_bouldin_index(features, labels):
    """
    Compute the Davies-Bouldin Index using scikit-learn.

    Parameters:
    - features: Array of feature vectors.
    - labels: Array of cluster labels.

    Returns:
    - db_index: Davies-Bouldin Index.
    """
    if len(set(labels)) <= 1:
        return np.inf  # Undefined for 1 or 0 clusters
    db_index = davies_bouldin_score(features, labels)
    return db_index


def compute_calinski_harabasz_index(features, labels):
    """
    Compute the Calinski-Harabasz Index.

    Parameters:
    - features: Array of feature vectors.
    - labels: Array of cluster labels.

    Returns:
    - ch_index: Calinski-Harabasz Index.
    """
    if len(set(labels)) <= 1:
        return 0  # Undefined for 1 or 0 clusters
    ch_index = calinski_harabasz_score(features, labels)
    return ch_index


def compute_transition_count(labels):
    """
    Compute the number of transitions between clusters.

    Parameters:
    - labels: Array of cluster labels.

    Returns:
    - transition_count: Number of transitions between different clusters.
    """
    transitions = np.sum(labels[:-1] != labels[1:])
    return transitions


def compute_average_segment_length(labels):
    """
    Compute the average length of contiguous segments.

    Parameters:
    - labels: Array of cluster labels.

    Returns:
    - average_length: Average segment length.
    """
    if len(labels) == 0:
        return 0
    changes = np.where(labels[:-1] != labels[1:])[0] + 1
    segments = np.split(labels, changes)
    segment_lengths = [len(segment) for segment in segments]
    average_length = np.mean(segment_lengths) if segment_lengths else 0
    return average_length


# -----------------------------------------
# Visualization Functions
# -----------------------------------------

def visualize_data(data, labels, time_steps, title, output_dir, filename_suffix):
    """
    General function to visualize data using t-SNE and create 2D plots.

    Parameters:
    - data: The feature data to visualize.
    - labels: Cluster labels for the data.
    - time_steps: Time steps or frame indices corresponding to the data.
    - title: Title for the plots.
    - output_dir: Directory to save plots.
    - filename_suffix: Suffix to add to output filenames.
    """
    # Prepare TSNE data
    tsne = TSNE(n_components=2, random_state=42)
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
    color_map[-1] = 'black'  # Assign black color for noise

    # Map labels to colors
    plot_data['color'] = plot_data['cluster'].map(color_map)

    # ----- 2D TSNE Plot -----
    plt.figure(figsize=(12, 10))
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
    print(f"t-SNE plot saved to {output_path_2d}")
    plt.close()


def visualize_full_fps(minCl, minSpl, sigma, labels, transformed_data, raw_features, name, fps, model_name, output_dir):
    """
    Visualize the full FPS data with clusters.

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
        filename_suffix=f"full_fps_clusters_transformed_{name}_sigma{sigma}_minCl{minCl}_minSpl{minSpl}"
    )


# -----------------------------------------
# Representative Extraction Functions
# -----------------------------------------

def extract_representative_points(labels, probabilities, data_indices):
    """
    Extract the indices of representative points for each cluster.

    Parameters:
    - labels: Cluster labels for each data point.
    - probabilities: Membership probabilities for each data point.
    - data_indices: Original indices of the data points (e.g., frame numbers).

    Returns:
    - representatives: Dictionary mapping cluster labels to representative point indices.
    """
    unique_clusters = set(labels)
    unique_clusters.discard(-1)  # Exclude noise

    representatives = {}
    for cluster in unique_clusters:
        cluster_indices = np.where(labels == cluster)[0]
        cluster_probabilities = probabilities[cluster_indices]
        # Find the index of the point with the highest probability in this cluster
        max_prob_index = cluster_indices[np.argmax(cluster_probabilities)]
        # Map to original data index if necessary
        representatives[cluster] = data_indices[max_prob_index]
    return representatives


def save_representative_images(representatives, dataset_samples, representatives_dir, sigma, minCl, minSpl):
    """
    Save representative images for each cluster.

    Parameters:
    - representatives: Dictionary mapping cluster labels to representative point indices.
    - dataset_samples: List of tuples (image_path, class_label).
    - representatives_dir: Directory to save representative images.
    - sigma: Sigma value used in Gaussian smoothing.
    - minCl: Minimum cluster size used in HDBSCAN.
    - minSpl: Minimum samples used in HDBSCAN.
    """
    os.makedirs(representatives_dir, exist_ok=True)

    for cluster_label, data_index in representatives.items():
        src_path = dataset_samples[data_index][0]  # Get the image path
        cluster_dir = os.path.join(representatives_dir, f"Cluster_{cluster_label}")
        os.makedirs(cluster_dir, exist_ok=True)
        dest_path = os.path.join(cluster_dir, f"representative_sigma{sigma}_minCl{minCl}_minSpl{minSpl}.png")
        shutil.copy(src_path, dest_path)
        print(f"Representative image for Cluster {cluster_label} saved to {dest_path}")

    # Handle noise cluster (-1) if present
    if -1 in representatives:
        cluster_label = -1
        data_index = representatives[cluster_label]
        src_path = dataset_samples[data_index][0]
        cluster_dir = os.path.join(representatives_dir, f"Noise")
        os.makedirs(cluster_dir, exist_ok=True)
        dest_path = os.path.join(cluster_dir, f"representative_sigma{sigma}_minCl{minCl}_minSpl{minSpl}.png")
        shutil.copy(src_path, dest_path)
        print(f"Representative image for Noise saved to {dest_path}")


# -----------------------------------------
# Main Processing Function
# -----------------------------------------

def process_combination(minCl, minSpl, params):
    """
    Process a single combination of minCl and minSpl.

    Parameters:
    - minCl: Minimum cluster size for HDBSCAN.
    - minSpl: Minimum samples for HDBSCAN.
    - params: Dictionary containing all necessary parameters.

    Returns:
    - metrics: Dictionary of evaluation metrics.
    """
    sigma = params['sigma']
    data = params['memmap_data']
    name = params['name']
    model_name = params['model_name']
    fps = params['fps']
    evaluate = params['evaluate']
    draw_plots = params['draw_plots']
    output_dir = params['output_dir']
    transformed_data = params['transformed_data']
    raw_features = params['raw_features']
    dataset_samples = params['dataset_samples']

    print(f"Processing minCl: {minCl}, minSpl: {minSpl} for sigma: {sigma}")

    # Create a new instance of HDBSCAN for each combination
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=minCl,
        min_samples=minSpl,
        memory=Memory(location="./dumps/cache", verbose=0)
    )
    hdbscan_clusterer.fit(data)
    labels = hdbscan_clusterer.labels_
    probabilities = hdbscan_clusterer.probabilities_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found: {num_clusters}")

    # Extract representative points for HDBSCAN clusters
    data_indices = np.arange(data.shape[0])  # Assuming data points are ordered
    representatives = extract_representative_points(labels, probabilities, data_indices)

    # Save representatives dictionary
    representatives_dir = os.path.join(output_dir, 'representatives')
    os.makedirs(representatives_dir, exist_ok=True)
    representatives_file = os.path.join(representatives_dir, f'representatives_sigma{sigma}_minCl{minCl}_minSpl{minSpl}.npy')
    np.save(representatives_file, representatives)
    print(f"Representatives saved to {representatives_file}")

    # Save representative images for HDBSCAN clusters
    save_representative_images(representatives, dataset_samples, representatives_dir, sigma, minCl, minSpl)

    metrics = {}
    if evaluate:
        # Compute evaluation metrics for HDBSCAN clustering
        temporal_purity = temporal_purity_score(labels)
        silhouette_avg = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
        inter_dist, intr_dist = compute_cluster_distances(data, labels)
        intra_var = compute_intra_cluster_variance(data, labels)
        db_index = compute_davies_bouldin_index(data, labels)
        ch_index = compute_calinski_harabasz_index(data, labels)
        transition_count = compute_transition_count(labels)
        average_segment_length = compute_average_segment_length(labels)
        # dunn_index = compute_dunn_index(data, labels)  # Uncomment if Dunn Index is implemented

        metrics_hdbscan = {
            'method': 'HDBSCAN',
            'sigma': sigma,
            'minCl': minCl,
            'minSpl': minSpl,
            'temporal_purity': temporal_purity,
            'silhouette_score': silhouette_avg,
            'davies_bouldin_index': db_index,
            'calinski_harabasz_index': ch_index,
            'intercluster_distance': inter_dist,
            'intracluster_distance': intr_dist,
            'intra_cluster_variance': intra_var,
            'transition_count': transition_count,
            'average_segment_length': average_segment_length
            # 'dunn_index': dunn_index,  # Uncomment if Dunn Index is implemented
        }

        # Perform random segmentation
        random_labels = perform_random_segmentation(data.shape[0], num_clusters, random_state=42)

        # Compute evaluation metrics for random segmentation
        temporal_purity_rand = temporal_purity_score(random_labels)
        silhouette_avg_rand = silhouette_score(data, random_labels) if len(set(random_labels)) > 1 else -1
        inter_dist_rand, intra_dist_rand = compute_cluster_distances(data, random_labels)
        intra_var_rand = compute_intra_cluster_variance(data, random_labels)
        db_index_rand = compute_davies_bouldin_index(data, random_labels)
        ch_index_rand = compute_calinski_harabasz_index(data, random_labels)
        transition_count_rand = compute_transition_count(random_labels)
        average_segment_length_rand = compute_average_segment_length(random_labels)
        # dunn_index_rand = compute_dunn_index(data, random_labels)  # Uncomment if Dunn Index is implemented

        metrics_random = {
            'method': 'Random_Segmentation',
            'sigma': sigma,
            'minCl': minCl,
            'minSpl': minSpl,
            'temporal_purity': temporal_purity_rand,
            'silhouette_score': silhouette_avg_rand,
            'davies_bouldin_index': db_index_rand,
            'calinski_harabasz_index': ch_index_rand,
            'intercluster_distance': inter_dist_rand,
            'intracluster_distance': intra_dist_rand,
            'intra_cluster_variance': intra_var_rand,
            'transition_count': transition_count_rand,
            'average_segment_length': average_segment_length_rand
            # 'dunn_index': dunn_index_rand,  # Uncomment if Dunn Index is implemented
        }

        # Extract representative points for Random Segmentation
        # Since random segmentation has contiguous segments, selecting the first frame as representative
        random_representatives = {}
        unique_random_clusters = set(random_labels)
        unique_random_clusters.discard(-1)  # Exclude noise if present
        for cluster in unique_random_clusters:
            cluster_indices = np.where(random_labels == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            representative_index = cluster_indices[0]  # First frame in the cluster
            random_representatives[cluster] = data_indices[representative_index]

        # Save representatives dictionary for Random Segmentation
        random_representatives_file = os.path.join(representatives_dir, f'random_representatives_sigma{sigma}_minCl{minCl}_minSpl{minSpl}.npy')
        np.save(random_representatives_file, random_representatives)
        print(f"Random segmentation representatives saved to {random_representatives_file}")

        # Save representative images for Random Segmentation
        save_representative_images(random_representatives, dataset_samples, representatives_dir, sigma, minCl, minSpl)

        # Combine metrics
        metrics = {'HDBSCAN': metrics_hdbscan, 'Random_Segmentation': metrics_random}

    if draw_plots:
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
            output_dir=output_dir
        )

    return metrics  # Return the metrics for this combination


# -----------------------------------------
# WCECluster Class
# -----------------------------------------

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
            feature_generator = FeatureGenerator(model_name=backbone, student=self.student)
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
            'raw_features': features,              # For potential future use
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