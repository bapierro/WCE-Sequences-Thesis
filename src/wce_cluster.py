import os
import shutil
from joblib import Memory
import torch
import torchvision.transforms as transforms
import pandas as pd
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

from model_name import Model
from feature_generator import FeatureGenerator

class WCECluster:
    def __init__(
        self, 
        dataset_path: str, 
        minCl: list[int] = [30], 
        minSpl: list[int] = [1], 
        batch_size: int = 32, 
        img_size: int = 224, 
        backbone = Model.ENDO_FM,  
        evaluate: bool = False,
        smooth= True, 
        save_clusters: bool = False, 
        output_dir = "./dumps/clustered_images",
        plot_cluster : bool = True,
        plot_time_series = False,
        student = True,
        auto_sigma=False,
        sigmas: list[float] = [6],
        fps = None,
        save_full_fps = False
    ):
        if fps is None:
            raise RuntimeError("Please specify FPS")
        
        self.sigmas = sigmas
        self.fps = fps
        self.save_full_fps = save_full_fps  # Whether to save and visualize the full FPS version
        self.plot_cluster = plot_cluster
        self.plot_time_series = plot_time_series
        self.minCl_values = minCl
        self.minSpl_values = minSpl
        self.evaluate = evaluate
        self.smooth = smooth
        self.save_clusters = save_clusters
        self.name = os.path.basename(dataset_path.rstrip('/'))
        self.auto_sigma = auto_sigma
        self.model_name = backbone.name
        self.backbone = FeatureGenerator(model_name=backbone, student=student)  # Adjust accordingly
        preprocess = v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5929, 0.3667, 0.1843],[0.1932, 0.1411, 0.0940]),
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
        feature_dir = f'./dumps/Features/{self.name}'
        os.makedirs(feature_dir, exist_ok=True)
        feature_file = os.path.join(feature_dir, f'{self.model_name}_{self.fps}FPS_features.npy')
        label_file = os.path.join(feature_dir, f'{self.model_name}_{self.fps}FPS_labels.npy')
        
        if os.path.exists(feature_file) and os.path.exists(label_file):
            print(f"Loading features from {feature_file}")
            self.features = np.load(feature_file)
            self.control_labels = np.load(label_file)
        else:
            transformed_data = []
            control_labels = []
            print("Beginning feature extraction...")
            
            # First pass: Collect all features and labels
            for img_batch, label_batch in tqdm(self.data_loader, desc="Passing images through backbone", unit="batch"):
                # Generate features using the backbone model
                features = self.backbone.generate(img_batch)
                features = features.view(features.size(0), -1)
                features_np = features.cpu().numpy()
                
                # Append features and labels to the list without smoothing
                transformed_data.append(features_np)
                control_labels.append(label_batch.cpu().numpy())
            
            # Stack the transformed data and control labels into final arrays
            self.features = np.vstack(transformed_data)
            self.control_labels = np.hstack(control_labels)
            
            # Save features and labels for future use
            np.save(feature_file, self.features)
            np.save(label_file, self.control_labels)
            print(f"Features saved to {feature_file}")
        

    def _cluster(self, minCl, minSpl):
        print(f"Clustering with minCl: {minCl}, minSpl: {minSpl}")
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=minCl, min_samples=minSpl, memory=Memory(location="./dumps/cache"))
        self.hdbscan.fit(self.transformed_data)

    def _extrapolate_clusters(self):
        """
        Extrapolate the cluster labels to the full dataset using k-NN.
        """
        # Use k-NN to predict labels for the full dataset
        knn = KNeighborsClassifier(n_neighbors=1)
        # Fit on transformed data
        print(f"Extrapolating clusters: Fitting KNN on data of shape {self.transformed_data.shape}")
        knn.fit(self.transformed_data, self.hdbscan.labels_)
        # Predict on full dataset
        print(f"Predicting labels for full data of shape {self.full_transformed_data.shape}")
        self.full_predicted_labels = knn.predict(self.full_transformed_data)
        print(f"Full predicted labels assigned with shape {self.full_predicted_labels.shape}")

    def _visualize_data(self, data, labels, time_steps, title, output_dir, filename_suffix, minCl=None, minSpl=None, sigma=None, control_labels=None):
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
        if control_labels is not None:
            sns.scatterplot(
                x=plot_data['tsne_1'],
                y=plot_data['tsne_2'],
                hue=[self.dataset.classes[label] for label in control_labels],
                palette=sns.color_palette("colorblind", n_colors=len(self.dataset.classes)),
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

    def _visualize(self, minCl, minSpl, sigma):
        fps_folder = f"{self.fps}FPS"
        output_dir = f"./dumps/Plots/{self.name}/{fps_folder}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"tsne_control_labels_{self.name}.svg")
        if os.path.exists(output_path):
            print(f"Plots exist for this combination: {output_path}")
            return
        print(f"Visualizing with minCl: {minCl}, minSpl: {minSpl}, sigma: {sigma}")
        
        # Time steps (assuming each point is a consecutive frame)
        time_steps = np.arange(len(self.hdbscan.labels_))
        
        # Visualize control labels
        self._visualize_data(
            data=self.transformed_data,
            labels=self.control_labels,
            time_steps=time_steps,
            title=f"{self.name} {fps_folder}",
            output_dir=output_dir,
            filename_suffix=f"control_labels_{self.name}",
            control_labels=self.control_labels
        )

        # Visualize cluster results
        self._visualize_data(
            data=self.transformed_data,
            labels=self.hdbscan.labels_,
            time_steps=time_steps,
            title=f"Clusters {self.name} (minCl={minCl}, minSpl={minSpl}, sigma={sigma})",
            output_dir=output_dir,
            filename_suffix=f"clusters_{self.name}_sigma{sigma}_minCl{minCl}_minSpl{minSpl}",
            minCl=minCl,
            minSpl=minSpl,
            sigma=sigma
        )

    def _visualize_original_distribution(self, sigma):
        """
        Visualize the original distribution with raw and smoothed features using original labels.
        This is saved only once per model and FPS.
        """
        fps_folder = f"{self.fps}FPS"
        output_dir_base = f"./dumps/Plots/{self.name}/{fps_folder}/{self.model_name}/FullFPSClusters"
        os.makedirs(output_dir_base, exist_ok=True)

        # Check if the plots already exist
        output_path_raw = os.path.join(output_dir_base, f"tsne_original_labels_raw_{self.name}_sigma{sigma}.svg")
        output_path_smoothed = os.path.join(output_dir_base, f"tsne_original_labels_smoothed_{self.name}_sigma{sigma}.svg")
        if os.path.exists(output_path_raw) and os.path.exists(output_path_smoothed):
            print(f"Original distribution plots already exist: {output_path_raw}")
            return

        print(f"Visualizing original distribution with sigma={sigma}")

        # Time steps
        time_steps = np.arange(len(self.full_control_labels))

        # Visualize raw features with original labels
        self._visualize_data(
            data=self.raw_features,
            labels=self.full_control_labels,
            time_steps=time_steps,
            title=f"{self.name}",
            output_dir=output_dir_base,
            filename_suffix=f"original_labels_raw_{self.name}_sigma{sigma}",
            control_labels=self.full_control_labels
        )

        # Visualize smoothed features with original labels
        self._visualize_data(
            data=self.full_transformed_data,
            labels=self.full_control_labels,
            time_steps=time_steps,
            title=f"{self.name} Smoothed (sigma={sigma})",
            output_dir=output_dir_base,
            filename_suffix=f"original_labels_smoothed_{self.name}_sigma{sigma}",
            control_labels=self.full_control_labels
        )

    def _visualize_full_fps(self, minCl, minSpl, sigma):
        """
        Visualize the full FPS data with clusters.
        This function will generate plots for both transformed and raw features.
        """
        fps_folder = f"{self.fps}FPS"
        output_dir_base = f"./dumps/Plots/{self.name}/{fps_folder}/{self.model_name}/FullFPSClusters"
        os.makedirs(output_dir_base, exist_ok=True)

        # Time steps
        time_steps = np.arange(len(self.full_predicted_labels))

        # Visualize full FPS clusters on transformed features
        self._visualize_data(
            data=self.full_transformed_data,
            labels=self.full_predicted_labels,
            time_steps=time_steps,
            title=f"{self.name} Smoothed (minCl={minCl}, minSpl={minSpl}, sigma={sigma})",
            output_dir=output_dir_base,
            filename_suffix=f"full_fps_clusters_transformed_{self.name}_sigma{sigma}_minCl{minCl}_minSpl{minSpl}",
            minCl=minCl,
            minSpl=minSpl,
            sigma=sigma
        )

        # Visualize full FPS clusters on raw features
        self._visualize_data(
            data=self.raw_features,
            labels=self.full_predicted_labels,
            time_steps=time_steps,
            title=f"Clusters {self.name} (minCl={minCl}, minSpl={minSpl}, sigma={sigma})",
            output_dir=output_dir_base,
            filename_suffix=f"full_fps_clusters_raw_{self.name}_sigma{sigma}_minCl{minCl}_minSpl{minSpl}",
            minCl=minCl,
            minSpl=minSpl,
            sigma=sigma
        )

    def _calculate_accuracy(self, true_labels, predicted_labels):
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

        return accuracy_score(true_labels, label_mapping)

    def evaluate_clustering(self, full=False):
        if full:
            true_labels = self.full_control_labels
            predicted_labels = self.full_predicted_labels
        else:
            true_labels = self.control_labels
            predicted_labels = self.hdbscan.labels_
        
        accuracy_with_anomalies = self._calculate_accuracy(true_labels, predicted_labels)
        nmi_with_anomalies = normalized_mutual_info_score(true_labels, predicted_labels)
        ari_with_anomalies = adjusted_rand_score(true_labels, predicted_labels)
        
        mask = predicted_labels != -1
        accuracy_without_anomalies = self._calculate_accuracy(true_labels[mask], predicted_labels[mask])
        nmi_without_anomalies = normalized_mutual_info_score(true_labels[mask], predicted_labels[mask])
        ari_without_anomalies = adjusted_rand_score(true_labels[mask], predicted_labels[mask])
        
        # Print statements can be kept or removed based on preference
        print("Evaluation with anomalies (-1 included):")
        print(f"Accuracy: {accuracy_with_anomalies:.4f}")
        print(f"NMI: {nmi_with_anomalies:.4f}")
        print(f"ARI: {ari_with_anomalies:.4f}\n")
        
        print("Evaluation without anomalies (-1 excluded):")
        print(f"Accuracy: {accuracy_without_anomalies:.4f}")
        print(f"NMI: {nmi_without_anomalies:.4f}")
        print(f"ARI: {ari_without_anomalies:.4f}")
        
        # Return the metrics in a dictionary
        return {
            'accuracy_with_anomalies': accuracy_with_anomalies,
            'nmi_with_anomalies': nmi_with_anomalies,
            'ari_with_anomalies': ari_with_anomalies,
            'accuracy_without_anomalies': accuracy_without_anomalies,
            'nmi_without_anomalies': nmi_without_anomalies,
            'ari_without_anomalies': ari_without_anomalies
        }

    def _copy_images_to_clusters(self, minCl, minSpl, sigma):
        """
        Copy images to directories based on cluster labels.
        """
        if not self.save_clusters:
            return
        
        fps_folder = f"{self.fps}FPS"
        out_dir = f"{self.output_dir}/{self.name}/{fps_folder}/{self.model_name}_sigma{sigma}_{minCl}_{minSpl}"
        # Create the output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        
        # Iterate through each image and copy it to the corresponding cluster directory
        for i, label in enumerate(self.full_predicted_labels):
            cluster_dir = os.path.join(out_dir, f"cluster_{label}" if label != -1 else "noise")
            
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir)
            
            # Get the original image path
            image_path, _ = self.dataset.samples[i]
            
            # Copy the image to the corresponding cluster directory
            shutil.copy(image_path, cluster_dir)

    def apply(self):
        self._extract_features()
        
        # Store raw features and original labels before any modifications
        self.raw_features = self.features.copy()
        self.full_control_labels = self.control_labels.copy()
        
        for sigma in self.sigmas:
            
            radius = int(3*sigma)
            transformed_data = gaussian_filter(
                self.features, sigma=sigma, radius=radius)
            
            self.transformed_data = transformed_data  # Assign transformed data to self.transformed_data
            
            # Store full transformed data
            self.full_transformed_data = self.transformed_data.copy()
            
            # Visualize the original distribution with original labels (only once)
            self._visualize_original_distribution(sigma)
            
            # Initialize a list to store evaluation results
            evaluation_results = []
            
            # Iterate over all combinations of minCl and minSpl
            for minCl, minSpl in product(self.minCl_values, self.minSpl_values):
                self._cluster(minCl, minSpl)
                self._visualize(minCl, minSpl, sigma)
                
                # Extrapolate clusters to full dataset
                self._extrapolate_clusters()
                
                # Save the full control labels
                labels_dir_full = f'./dumps/Labels/{self.name}/{self.fps}FPS'
                os.makedirs(labels_dir_full, exist_ok=True)
                full_control_labels_path = os.path.join(labels_dir_full, f"full_control_labels_{self.name}.npy")
                np.save(full_control_labels_path, self.full_control_labels)
                print(f"Control labels for full FPS data saved to {full_control_labels_path}")
                
                if self.evaluate:
                    # Get the evaluation metrics on the full dataset
                    metrics = self.evaluate_clustering(full=True)
                    
                    # Add the current parameters to the metrics
                    metrics['sigma'] = sigma
                    metrics['minCl'] = minCl
                    metrics['minSpl'] = minSpl
                    
                    # Append the metrics to the evaluation_results list
                    evaluation_results.append(metrics)
                
                if self.save_full_fps:
                    self._visualize_full_fps(minCl, minSpl, sigma)
                
                if self.save_clusters:
                    self._copy_images_to_clusters(minCl, minSpl, sigma)
            
            # After all computations, create a DataFrame and save or display it
            if self.evaluate and evaluation_results:
                results_df = pd.DataFrame(evaluation_results)
                print("\nEvaluation Results:")
                print(results_df)
                
                # Save the results to a CSV file
                eval_dir = f"./dumps/Evaluation/{self.name}/{self.model_name}/Sigma{sigma}"
                os.makedirs(eval_dir, exist_ok=True)
                output_path = os.path.join(eval_dir, f"evaluation_results_{self.name}.csv")
                results_df.to_csv(output_path, index=False)
                print(f"Evaluation results saved to {output_path}")