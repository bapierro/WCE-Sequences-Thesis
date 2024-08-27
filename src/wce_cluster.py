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


class WCECluster:
    def __init__(self,dataset_path : str, minCl : int | None = 30, minSpl : int | None = 1, batch_size : int = 32):
        self.minCl = minCl
        self.minSpl = minSpl
        
        self.backbone = FeatureGenerator(model_name=Model.ENDO_FM)
        preprocess = transforms.Compose([
            transforms.Resize(336),
            transforms.CenterCrop(336),
            transforms.ToTensor()
        ])
        self.dataset = datasets.ImageFolder(dataset_path, transform=preprocess)
        
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        self.data_loader = DataLoader(self.dataset,batch_size=batch_size)
        
    def _extract_features(self):
        transformed_data = []
        control_labels = []
        print("Beginning feature extraction...")
        for img_batch,label_batch in tqdm(self.data_loader, desc="Passing images through backbone", unit="batch"):
            features = self.backbone.generate(img_batch)
            features = features.view(features.size(0),-1)
            transformed_data.append(features.cpu().numpy())
            control_labels.append(label_batch.cpu().numpy())
            
            
        
        self.transformed_data = np.vstack(transformed_data)
        self.control_labels = np.hstack(control_labels)
        
    def _cluster(self):
        #min_cluster_values = [5,15,30,60,100,200,500] if self.minCl is None else [self.minCl]
        #min_samples_values = [1,5,15,30,60,100,200] if self.minSpl is None else [self.minSpl]
        
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=self.minCl,min_samples=self.minSpl,gen_min_span_tree=True)
        self.hdbscan.fit(self.transformed_data)
    
    def _visualize(self):
        figsize = (24, 10)  # Increase the width to accommodate both plots
        
        tsne = TSNE(n_components=2, random_state=20)
        vanilla_data = tsne.fit_transform(self.transformed_data)
        
        plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        sns.scatterplot(
            x=vanilla_data[:, 0], 
            y=vanilla_data[:, 1], 
            hue=[self.dataset.classes[label] for label in self.control_labels], 
            palette=sns.color_palette("hsv", len(self.dataset.classes)), 
            legend="full",
            ax=axs[0]
        )
        
        axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0., fontsize="small")
        axs[0].set_title("Control Labels")
        palette = sns.color_palette()
        # cluster_colors = [sns.desaturate(palette[col], sat)
        #                 if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
        #                 zip(self.hdbscan.labels_, self.hdbscan.probabilities_)]
        
        
        print(len(self.hdbscan.labels_))
        sns.scatterplot(
            x=vanilla_data[:, 0], 
            y=vanilla_data[:, 1], 
            hue=self.hdbscan.labels_, 
            palette=sns.color_palette("hsv", len(set(self.hdbscan.labels_))),
            legend='full',
            ax=axs[1]
        )
        
        axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0., fontsize="small")
        axs[1].set_title("Cluster Results")
        
        plt.tight_layout()        
        plt.show()

    def apply(self):
        self._extract_features()
        self._cluster()
        self._visualize()