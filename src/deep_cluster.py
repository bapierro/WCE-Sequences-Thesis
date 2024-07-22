import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from tqdm import tqdm
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from model_name import Model
from feature_generator import FeatureGenerator
from clustering_method import ClusterMethod

class DeepCluster:
    def __init__(self, model_name: Model, dataset_path: str, clustering_method=ClusterMethod.KMEANS, fraction: float = 0.1, batch_size: int = 32, n_clusters: int = 13):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.clustering_method = clustering_method
        self.fraction = fraction
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.feature_generator = FeatureGenerator(model_name, pretrained=True)
        
        if self.model_name == Model.ENDO_FM:
            self.preprocess = transforms.Compose([
                transforms.Resize(336),
                transforms.CenterCrop(336),
                transforms.ToTensor(),
            ])
        elif self.model_name in (Model.RES_NET_18, Model.RES_NET_34, Model.RES_NET_50, Model.RES_NET_152, Model.RES_NET_101):
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.dataset = datasets.ImageFolder(self.dataset_path, transform=self.preprocess)
        self.subset_dataset = self.get_subset(self.dataset, self.fraction)
        self.data_loader = DataLoader(self.subset_dataset, batch_size=self.batch_size, shuffle=False)

    def get_subset(self, dataset, fraction):
        class_indices = {cls: [] for cls in range(len(dataset.classes))}
        for idx, (path, class_idx) in enumerate(dataset.samples):
            class_indices[class_idx].append(idx)

        subset_indices = []
        for class_idx, indices in class_indices.items():
            subset_size = max(1, int(len(indices) * fraction))
            subset_indices.extend(random.sample(indices, subset_size))

        self.subset_indices = subset_indices
        return Subset(dataset, subset_indices)

    def extract_features(self):
        all_features = []
        all_labels = []

        print("Extracting features from images...")
        for img_batch, label_batch in tqdm(self.data_loader, desc="Feature Extraction", unit="batch"):
            features = self.feature_generator.generate(img_batch)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

        self.all_features = np.vstack(all_features)
        self.all_labels = np.hstack(all_labels)

    def perform_clustering(self):
        self.labels_list = []
        self.params_list = []
        
        match self.clustering_method:
            case ClusterMethod.KMEANS:
                print("Performing KMeans clustering...")
                self.kmeans = KMeans(n_clusters=self.n_clusters)
                self.labels = self.kmeans.fit_predict(self.all_features)
                self.labels_list.append(self.labels)
                self.params_list.append({'method': 'KMeans', 'n_clusters': self.n_clusters})
            case ClusterMethod.DBSCAN:
                print("Performing DBSCAN clustering with multiple hyperparameters...")
                eps_values = [1.5, 2, 3, 4, 5, 10]
                min_samples_values = [5, 10, 15]
                for eps in eps_values:
                    for min_samples in min_samples_values:
                        print(f"  Trying eps={eps}, min_samples={min_samples}")
                        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = self.dbscan.fit_predict(self.all_features)
                        self.labels_list.append(labels)
                        self.params_list.append({'method': 'DBSCAN', 'eps': eps, 'min_samples': min_samples})
            case ClusterMethod.HDBSCAN:
                print("Performing HDBSCAN clustering with multiple hyperparameters...")
                min_cluster_sizes = [5,15,30,60,100]
                min_samples_list = [1,5,15,30,60,100]
                for min_cluster_size in min_cluster_sizes:
                    for min_samples in min_samples_list:
                        print(f"  Trying min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                        labels = self.hdbscan.fit_predict(self.all_features)
                        self.labels_list.append(labels)
                        self.params_list.append({'method': 'HDBSCAN', 'min_cluster_size': min_cluster_size, 'min_samples': min_samples})

    def evaluate_clustering(self):
        best_nmi = -1
        best_labels = None
        best_params = None
        
        for i, labels in enumerate(self.labels_list):
            print(f"Evaluating clustering performance for params: {self.params_list[i]}")
            nmi = normalized_mutual_info_score(self.all_labels, labels)
            ari = adjusted_rand_score(self.all_labels, labels)
            print(f'  Normalized Mutual Information (NMI): {nmi:.4f}')
            print(f'  Adjusted Rand Index (ARI): {ari:.4f}')
            
            if nmi > best_nmi:
                best_nmi = nmi
                best_labels = labels
                best_params = self.params_list[i]
        
        self.best_nmi = best_nmi
        self.best_labels = best_labels
        self.best_params = best_params
        
        print(f'Best NMI: {self.best_nmi:.4f} with params: {self.best_params}')

    def save_cluster_images(self):
        name = self.dataset_path.split("/")[-1]
        cluster_dir = f"./dumps/{name}_clusters_{self.best_params['method']}"
        os.makedirs(cluster_dir, exist_ok=True)
        for cluster in set(self.best_labels):
            cluster_path = os.path.join(cluster_dir, f"cluster_{cluster}")
            os.makedirs(cluster_path, exist_ok=True)
            cluster_indices = [idx for idx, label in zip(self.subset_indices, self.best_labels) if label == cluster]
            for idx in cluster_indices:
                src_path = self.dataset.samples[idx][0]
                #print(f"Copying {src_path} to {cluster_path}") # zum debuggen
                shutil.copy(src_path, cluster_path)
        print(f"Saved images for clusters with params: {self.best_params} to {cluster_dir}")

    def visualize(self):
        print("Performing t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_features = tsne.fit_transform(self.all_features)

        plt.figure(figsize=(16, 7))

        plt.subplot(1, 2, 1)
        sns.scatterplot(
            x=tsne_features[:, 0], 
            y=tsne_features[:, 1], 
            hue=[self.dataset.classes[label] for label in self.all_labels], 
            palette=sns.color_palette("hsv", len(self.dataset.classes)), 
            legend='full'
        )
        plt.title('t-SNE Visualization of True Labels')
        plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

        best_labels_unique = list(set(self.best_labels))
        best_labels_unique.sort()  
        best_palette = sns.color_palette("hsv", len(best_labels_unique))
        
        if -1 in best_labels_unique:
            best_palette = ['#000000' if label == -1 else color for label, color in zip(best_labels_unique, best_palette)]

        plt.subplot(1, 2, 2)
        sns.scatterplot(
            x=tsne_features[:, 0], 
            y=tsne_features[:, 1], 
            hue=self.best_labels, 
            palette=best_palette, 
            legend='full'
        )
        plt.title(f't-SNE Visualization of {self.best_params} Clusters')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    def run_experiment(self):
        self.extract_features()
        self.perform_clustering()
        self.evaluate_clustering()
        self.save_cluster_images()
        self.visualize()