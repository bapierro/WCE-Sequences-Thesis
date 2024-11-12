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
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
# import umap
from model_name import Model
from feature_generator import FeatureGenerator
import pandas as pd
from datetime import datetime

# Function to generate a unique run ID based on the current timestamp
def generate_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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


def density_based_clustering_validation_index(features,labels):
    if len(set(labels)) <= 1:
        return 0 
    dbcv_index = hdbscan.validity.validity_index(X=features.astype(np.float64),labels=labels,d=768)
    return dbcv_index

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


def visualize_full_fps(minCl, minSpl, sigma, labels, transformed_data, raw_features, name, fps, model_name, output_dir, original_labels=None, class_names=None):
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
    - original_labels: Original class labels from the dataset (if available).
    """
    fps_folder = f"{fps}FPS"
    output_dir_base = os.path.join(output_dir, name, fps_folder, model_name, "FullFPSClusters")
    os.makedirs(output_dir_base, exist_ok=True)

    # Time steps
    time_steps = np.arange(len(labels))

    # Prepare t-SNE data
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(transformed_data)

    if original_labels is not None:
        # Map numerical class labels to class names
        original_label_names = [class_names[label] for label in original_labels]


        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'time': time_steps,
            'tsne_1': tsne_data[:, 0],
            'tsne_2': tsne_data[:, 1],
            'cluster_labels': labels,
            'original_labels': original_label_names
        })

        # Generate color maps
        colors = px.colors.qualitative.G10

        # Color map for original labels (ground truth)
        unique_original_labels = np.unique(original_label_names)
        color_map_original = {label: colors[i % len(colors)] for i, label in enumerate(unique_original_labels)}

        # Color map for cluster labels
        unique_cluster_labels = np.unique(labels)
        cluster_labels_no_noise = [label for label in unique_cluster_labels if label != -1]
        color_map_cluster = {label: colors[i % len(colors)] for i, label in enumerate(cluster_labels_no_noise)}
        color_map_cluster[-1] = 'black'  # Assign black color for noise

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        # Left subplot: Original labels (Ground Truth)
        sns.scatterplot(
            ax=axes[0],
            x=plot_data['tsne_1'],
            y=plot_data['tsne_2'],
            hue=plot_data['original_labels'],
            palette=color_map_original,
            legend="full",
            alpha=0.6
        )
        axes[0].set_title(f"{name} {model_name} Ground Truth")
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')

        # Right subplot: Cluster labels
        sns.scatterplot(
            ax=axes[1],
            x=plot_data['tsne_1'],
            y=plot_data['tsne_2'],
            hue=plot_data['cluster_labels'],
            palette=color_map_cluster,
            legend="full",
            alpha=0.6
        )
        axes[1].set_title(f"{name} {model_name} Clustering (minCl={minCl}, minSpl={minSpl})")
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')

        # Adjust legends for Ground Truth
        handles_original, labels_original = axes[0].get_legend_handles_labels()
        by_label_original = dict(zip(labels_original, handles_original))
        axes[0].legend(by_label_original.values(), by_label_original.keys(), loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0., fontsize="small")

        # Adjust legends for Clustering
        handles_cluster, labels_cluster = axes[1].get_legend_handles_labels()
        by_label_cluster = dict(zip(labels_cluster, handles_cluster))
        axes[1].legend(by_label_cluster.values(), by_label_cluster.keys(), loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0., fontsize="small")

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_dir_base, f"tsne_dual_{name}_sigma{sigma}_minCl{minCl}_minSpl{minSpl}.svg")
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"Dual t-SNE plot saved to {output_path}")
        plt.close()
    else:
        # Existing code to visualize clusters with cluster labels
        visualize_data(
            data=transformed_data,
            labels=labels,
            time_steps=time_steps,
            title=f"{name} {model_name} (minCl={minCl}, minSpl={minSpl}, sigma={sigma})",
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

def evaluate_clustering(true_labels, predicted_labels):
    """
    Evaluate clustering performance metrics.

    Parameters:
    - true_labels: Ground truth labels.
    - predicted_labels: Labels predicted by the clustering algorithm.

    Returns:
    - A dictionary containing accuracy, NMI, and ARI.
    """

    # Map the predicted labels to true labels using the Hungarian algorithm
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)

    # Create a cost matrix for the Hungarian algorithm
    cost_matrix = np.zeros((len(unique_true), len(unique_pred)))
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            cost_matrix[i, j] = np.sum((true_labels == true_label) & (predicted_labels == pred_label))

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # negate cost_matrix to maximize

    # Create a mapping from predicted labels to true labels
    label_mapping = {}
    for i, j in zip(row_ind, col_ind):
        label_mapping[unique_pred[j]] = unique_true[i]

    # Map the predicted labels to the true labels
    mapped_predicted_labels = np.array([label_mapping.get(label, -1) for label in predicted_labels])

    # Compute metrics
    accuracy = accuracy_score(true_labels, mapped_predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    return {
        'accuracy': accuracy,
        'nmi': nmi,
        'ari': ari
    }


def fibonacci():
    a, b = 13, 21
    while True:
        yield a
        a, b = b, a + b


# -----------------------------------------
# WCECluster Class
# -----------------------------------------
class WCECluster:
    def __init__(
        self,
        dataset_path: str,
        minCls: list[int] = None,  # List of min_cluster_size values
        batch_size: int = 32,
        img_size: int = 224,
        backbones=None,  # Accept a list of models
        evaluate: bool = True,
        smooth=True,
        student=True,
        save_representatives=True,
        draw_plot = True,
        sigmas: list[float] = [6],
        fps=None,
        recompute=False,
        external_validation=False,  # Added external_validation parameter
        run_id= None,
        output_root: str = "./dumps", # Root directory for outputs
        reduce = False,
        cendo_backbone = None
    ):
        """
        Initialize the WCECluster class.

        Parameters:
        - dataset_path: Path to the dataset directory.
        - minCl: List of minimum cluster sizes for HDBSCAN.
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
        - external_validation: Whether to compute supervised evaluation metrics.
        """
        if fps is None:
            raise RuntimeError("Please specify FPS")
        if backbones is None or not isinstance(backbones, list):
            raise ValueError("Please provide a list of backbone models.")
        
        self.minCls = minCls
        self.reduce = reduce
        self.all_metrics = []  # List to store all evaluation metrics
        self.output_root = output_root  
        self.run_id = run_id if run_id is not None else generate_run_id()
        self.recompute = recompute
        self.sigmas = sigmas
        self.fps = fps
        self.evaluate = evaluate
        self.smooth = smooth
        self.backbones = backbones  # Now a list of models
        self.student = student
        self.batch_size = batch_size
        self.img_size = img_size
        self.name = os.path.basename(dataset_path.rstrip('/'))
        self.save_representatives = save_representatives
        self.draw_plot = draw_plot
        self.external_validation = external_validation  # Store external_validation
        self.cendo_backbone = cendo_backbone

        # Initialize DataLoader without modifying dataset based on FPS
        self.dataset = datasets.ImageFolder(dataset_path)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        self.pretty_model_names = {
            Model.CENDO_FM: "CendoFM",
            Model.ENDO_FM: "EndoFM",
            Model.RES_NET_50: "ResNet-50",
            Model.Swin_v2_B: "SwinV2-B"
            }

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
        elif backbone in (Model.RES_NET_50, Model.RES_NET_101, Model.RES_NET_18, Model.RES_NET_34, Model.RES_NET_152,Model.Swin_v2_B):
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
        feature_file = os.path.join(feature_dir, f'{backbone.name}_{30}FPS_features.npy')

        if os.path.exists(feature_file) and (not self.recompute or backbone != Model.CENDO_FM):
            print(f"Loading features from {feature_file} for model {backbone.name}")
            features = np.load(feature_file)
        else:
            # Initialize the feature generator
            feature_generator = FeatureGenerator(model_name=backbone, student=self.student,cb=self.cendo_backbone)
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
        """
        print(f"Processing sigma: {sigma} for model: {backbone_name}")
        
        # Apply Gaussian filter to features
        if self.smooth:
            transformed_data = gaussian_filter(features, sigma=sigma)
        else:
            transformed_data = features.copy()
            
        if self.reduce:
            transformed_data = umap.UMAP(n_neighbors=15,n_components=50, min_dist = 0.0).fit_transform(transformed_data)
        self.transformed_data = transformed_data  # Assign transformed data

        # Initialize HDBSCAN's Memory
        cache_dir = os.path.join(self.output_root, "cache", self.run_id)
        os.makedirs(cache_dir, exist_ok=True)
        memory = Memory(location=cache_dir, verbose=0)

        # Initialize variables to keep track of the best clustering
        best_eval_metric = -np.inf
        best_labels = None
        best_minCl = None
        patience = 7
        retries = patience
        validity_index = None

        # Iterate over different minCl values
        minCls = self.minCls if self.minCls else fibonacci()
        for minCl in minCls:
            print(f"Clustering with min_cluster_size={minCl}")
            # Initialize and fit HDBSCAN with min_samples=1
            hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=minCl,
                min_samples=1,
                metric='euclidean',
                memory=memory,
                core_dist_n_jobs=-1,
                prediction_data=True
            )
            hdbscan_clusterer.fit(transformed_data)
            # soft_clusters = hdbscan.all_points_membership_vectors(hdbscan_clusterer)
            # labels = np.array([np.argmax(x) for x in soft_clusters])
            labels = hdbscan_clusterer.labels_

            # Compute the metrics
            epsilon = 1e-10
            chi = compute_calinski_harabasz_index(transformed_data, labels)
            dbi = compute_davies_bouldin_index(transformed_data, labels)
            dbcv = density_based_clustering_validation_index(transformed_data, labels)

            # Adjust DBCV to be in the range [0, 2]
            dbcv_shifted = dbcv + 1

            # Compute the evaluation metric
            eval_metric = (chi / (dbi + epsilon)) * dbcv_shifted

            print(f"Evaluation Metric for minCl={minCl}: {eval_metric}")

            if eval_metric > best_eval_metric:
                best_eval_metric = eval_metric
                best_labels = labels
                best_minCl = minCl
                retries = patience
            else:
                retries -= 1
            
            if retries == 0:
                break

        if best_labels is None:
            print(f"No valid clustering found for sigma={sigma} with the provided minCl values.")
            return  # Exit if no valid clustering was found

        print(f"Best minCl selected: {best_minCl} with Evaluation Metric: {best_eval_metric}")
        labels = best_labels
        minCl = best_minCl
        probabilities = hdbscan_clusterer.probabilities_
        data_indices = np.arange(transformed_data.shape[0])  # Assuming data points are ordered
        representatives = extract_representative_points(labels, probabilities, data_indices)

        # Save representatives dictionary
        representatives_dir = os.path.join(self.output_root, 'representatives', self.run_id)
        os.makedirs(representatives_dir, exist_ok=True)
        representatives_file = os.path.join(representatives_dir, f'representatives_sigma{sigma}_minCl{minCl}_minSpl1.npy')
        np.save(representatives_file, representatives)
        print(f"Representatives saved to {representatives_file}")

        
        
        if self.draw_plot:
            
            visualize_full_fps(
                minCl=minCl,
                minSpl=1,  # Fixed value
                sigma=sigma,
                labels=labels,
                transformed_data=transformed_data,
                raw_features=features,
                name=self.name,
                fps=self.fps,
                model_name=backbone_name,
                output_dir=self.output_dir,
                original_labels=self.targets if self.external_validation else None,
                class_names=self.dataset.classes
            )


        # Save representative images for HDBSCAN clusters
        if self.save_representatives:
            save_representative_images(representatives, self.dataset_samples, representatives_dir, sigma, minCl, minSpl=1)

        metrics = {}
        if self.evaluate:
            # Compute evaluation metrics for HDBSCAN clustering
            temporal_purity = temporal_purity_score(labels)
            silhouette_avg = silhouette_score(transformed_data, labels) if len(set(labels)) > 1 else -1
            inter_dist, intr_dist = compute_cluster_distances(transformed_data, labels)
            intra_var = compute_intra_cluster_variance(transformed_data, labels)
            db_index = compute_davies_bouldin_index(transformed_data, labels)
            dbcv = density_based_clustering_validation_index(transformed_data,labels)
            eval_metric = compute_calinski_harabasz_index(transformed_data, labels)
            transition_count = compute_transition_count(labels)
            average_segment_length = compute_average_segment_length(labels)

            metrics_hdbscan = {
                'method': 'HDBSCAN',
                'sigma': sigma,
                'minCl': minCl,
                'minSpl': 1,  # Fixed value
                'Temporal Purity': temporal_purity,
                'Davies-Bouldin Index': db_index,
                'Calinski–Harabasz Index': eval_metric,
                'Intercluster Distance': inter_dist,
                'Intracluster Distance': intr_dist,
                'Density Based Clustering Validation Index': dbcv,
                'Intracluster Variance': intra_var,
                'Number of Transitions': transition_count,
                'Silhouette Coefficient': silhouette_avg,
                'Average Segment Length': average_segment_length,
                'model_name': backbone_name,
                'dataset_name': self.name,
            }

            # Compute supervised metrics if external_validation is True
            true_labels = np.array(self.targets)
            predicted_labels = labels

            # Exclude anomalies
            mask = predicted_labels != -1
            noise_percentage = 1 - (np.sum(mask) / len(true_labels))
            metrics_hdbscan.update({"Noise Percentage": noise_percentage})
            
            if self.external_validation:
                if np.any(mask):
                    # Use masked labels
                    true_labels_masked = true_labels[mask]
                    predicted_labels_masked = predicted_labels[mask]

                    # Compute the supervised metrics
                    supervised_metrics = evaluate_clustering(true_labels_masked, predicted_labels_masked)

                    # Update metrics dictionary
                    metrics_hdbscan.update(supervised_metrics)
                else:
                    print("All points are labeled as anomalies (-1). Cannot compute supervised metrics without anomalies.")
                    supervised_metrics = {
                        'accuracy': None,
                        'nmi': None,
                        'ari': None
                    }
                    metrics_hdbscan.update(supervised_metrics)

            # Perform random segmentation for comparison
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if num_clusters <= 0:
                print("No clusters found to perform random segmentation.")
                random_labels = np.full(transformed_data.shape[0], -1)
            else:
                random_labels = perform_random_segmentation(transformed_data.shape[0], num_clusters, random_state=42)

            # Compute evaluation metrics for random segmentation
            if len(set(random_labels)) <= 1 or (len(set(random_labels)) == 2 and -1 in random_labels):
                ch_index_rand = -1
                print(f"Calinski-Harabasz Index for Random Segmentation: Undefined (only one cluster or all noise). Assigned CH Index = {ch_index_rand}")
            else:
                ch_index_rand = calinski_harabasz_score(transformed_data, random_labels)
            
            
            

            if self.evaluate:
                # Compute other metrics only if clustering is valid
                temporal_purity_rand = temporal_purity_score(random_labels)
                inter_dist_rand, intra_dist_rand = compute_cluster_distances(transformed_data, random_labels)
                intra_var_rand = compute_intra_cluster_variance(transformed_data, random_labels)
                silhouette_avg_rand = silhouette_score(transformed_data, random_labels) if len(set(random_labels)) > 1 else -1
                dbcv_rand = density_based_clustering_validation_index(transformed_data,labels)
                db_index_rand = compute_davies_bouldin_index(transformed_data, random_labels)
                transition_count_rand = compute_transition_count(random_labels)
                average_segment_length_rand = compute_average_segment_length(random_labels)

                metrics_random = {
                    'method': 'Random Segmentation',
                    'sigma': sigma,
                    'minCl': minCl,
                    'minSpl': 1,
                    'Temporal Purity': temporal_purity_rand,
                    'Davies-Bouldin Index': db_index_rand,
                    'Calinski–Harabasz Index': ch_index_rand,
                    'Density Based Clustering Validation Index': dbcv_rand,
                    'Intercluster Distance': inter_dist_rand,
                    'Intracluster Distance': intra_dist_rand,
                    'Intracluster Variance': intra_var_rand,
                    'Number of Transitions': transition_count_rand,
                    'Silhouette Coefficient': silhouette_avg_rand,
                    'Average Segment Length': average_segment_length_rand,
                    'model_name': backbone_name,
                    'dataset_name': self.name,
                }

                # Compute supervised metrics for Random Segmentation if desired
                if self.external_validation and num_clusters > 0:
                    true_labels = np.array(self.targets)
                    predicted_labels = random_labels
                    mask_rand = predicted_labels != -1
                    if np.any(mask_rand):
                        true_labels_masked_rand = true_labels[mask_rand]
                        predicted_labels_masked_rand = predicted_labels[mask_rand]
                        supervised_metrics_rand = evaluate_clustering(true_labels_masked_rand, predicted_labels_masked_rand)
                        metrics_random.update(supervised_metrics_rand)
                    else:
                        print("All points are labeled as anomalies (-1) in Random Segmentation. Cannot compute supervised metrics.")
                        metrics_random.update({'accuracy': None, 'nmi': None, 'ari': None})

                # Extract representative points for Random Segmentation
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
                random_representatives_file = os.path.join(representatives_dir, f'random_representatives_sigma{sigma}_minCl{minCl}_minSpl1.npy')
                np.save(random_representatives_file, random_representatives)
                print(f"Random segmentation representatives saved to {random_representatives_file}")

                # Save representative images for Random Segmentation
                if self.save_representatives:
                    save_representative_images(random_representatives, self.dataset_samples, representatives_dir, sigma, minCl, minSpl=1)

                # Combine metrics
                metrics = {'HDBSCAN': metrics_hdbscan, 'Random_Segmentation': metrics_random}
            else:
                metrics = {'HDBSCAN': metrics_hdbscan}

            
            if self.evaluate and metrics:
                # Flatten metrics
                flattened_metrics = []
                for method_name, method_metrics in metrics.items():
                    flattened_metrics.append(method_metrics)

                # Create DataFrame from flattened_metrics
                combined_results_df = pd.DataFrame(flattened_metrics)
                print("\nCombined Evaluation Results:")
                print(combined_results_df)

                # Save the combined results to a CSV file
                eval_dir = os.path.join(
                    self.output_root, "Evaluation_Unsupervised", self.name, backbone_name, f"Sigma{sigma}"
                )
                os.makedirs(eval_dir, exist_ok=True)
                output_path = os.path.join(eval_dir, f"evaluation_results_combined_sigma{sigma}.csv")
                combined_results_df.to_csv(output_path, index=False)
                print(f"Combined evaluation results saved to {output_path}")
            else:
                flattened_metrics = []

            # Clean up temporary files
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                print(f"Could not delete cache directory {cache_dir}: {e}")

        
            return flattened_metrics


    def apply(self):
        for backbone in self.backbones:
            print(f"\nProcessing with model: {backbone.name}")
            self.model_name = self.pretty_model_names[backbone]

            features = self._extract_features(backbone)

            self.raw_features = features.copy()

            default_fps = 30  
            if self.fps != default_fps:
                frame_interval = int(round(default_fps / self.fps))
                indices = np.arange(0, len(features), frame_interval)
                features = features[indices]
                self.dataset_samples = [self.dataset.samples[i] for i in indices]
                self.targets = [self.dataset.targets[i] for i in indices]
            else:
                self.dataset_samples = self.dataset.samples
                self.targets = self.dataset.targets

            fps_folder = f"{self.fps}FPS"
            self.output_dir = os.path.join(
                self.output_root, "Plots", self.name, fps_folder, self.model_name, "FullFPSClusters"
            )
            os.makedirs(self.output_dir, exist_ok=True)

            # Process each sigma value
            for sigma in self.sigmas:
                flattened_metrics = self._process_sigma(sigma, features, self.model_name)
                if flattened_metrics:
                    self.all_metrics.extend(flattened_metrics)