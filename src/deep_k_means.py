import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import numpy as np
from tqdm import tqdm
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from model_name import Model
from feature_generator import FeatureGenerator

class DeepKMeans:
    def __init__(self, model_name: Model, dataset_path: str, fraction: float = 0.1, batch_size: int = 32, n_clusters: int = 13):
        self.model_name = model_name
        self.dataset_path = dataset_path
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
        elif self.model_name in (Model.RES_NET_18,Model.RES_NET_34,Model.RES_NET_50 ,Model.RES_NET_152,Model.RES_NET_101):
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
        print("Performing KMeans clustering...")
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans_labels = self.kmeans.fit_predict(self.all_features)

    def evaluate_clustering(self):
        print("Evaluating clustering performance...")
        self.nmi = normalized_mutual_info_score(self.all_labels, self.kmeans_labels)
        self.ari = adjusted_rand_score(self.all_labels, self.kmeans_labels)

        # Assign each cluster to the most frequent class, ensuring unique assignments
        class_counts_per_cluster = {}
        for cluster in range(self.n_clusters):
            cluster_indices = np.where(self.kmeans_labels == cluster)[0]
            class_counts = Counter(self.all_labels[cluster_indices])
            class_counts_per_cluster[cluster] = class_counts

        self.cluster_to_class = {}
        assigned_classes = set()
        for cluster, class_counts in sorted(class_counts_per_cluster.items(), key=lambda x: -sum(x[1].values())):
            for class_label, _ in class_counts.most_common():
                if class_label not in assigned_classes:
                    self.cluster_to_class[cluster] = class_label
                    assigned_classes.add(class_label)
                    break

        # Ensure all clusters are assigned
        all_classes = set(range(len(self.dataset.classes)))
        remaining_classes = list(all_classes - assigned_classes)
        for cluster in range(self.n_clusters):
            if cluster not in self.cluster_to_class:
                self.cluster_to_class[cluster] = remaining_classes.pop(0)

        # Predict labels based on cluster to class mapping
        self.predicted_labels = np.array([self.cluster_to_class[cluster] for cluster in self.kmeans_labels])

        # Calculate accuracy
        self.accuracy = accuracy_score(self.all_labels, self.predicted_labels)

        print(f'Normalized Mutual Information (NMI): {self.nmi:.4f}')
        print(f'Adjusted Rand Index (ARI): {self.ari:.4f}')
        print(f'Accuracy: {self.accuracy:.4f}')

    def visualize(self):
        print("Performing t-SNE visualization...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_features = tsne.fit_transform(self.all_features)

        plt.figure(figsize=(16, 7))

        # Plot true labels
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=[self.dataset.classes[label] for label in self.all_labels], palette=sns.color_palette("hsv", len(self.dataset.classes)), legend='full')
        plt.title('t-SNE Visualization of True Labels')
        plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot predicted labels based on cluster to class mapping
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=self.kmeans_labels, palette=sns.color_palette("hsv", self.n_clusters), legend='full')
        plt.title('t-SNE Visualization of KMeans Clusters')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    def run_experiment(self):
        self.extract_features()
        self.perform_clustering()
        self.evaluate_clustering()
        self.visualize()