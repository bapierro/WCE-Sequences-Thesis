import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm

from feature_generator import FeatureGenerator  # Adjust import path
from model_name import Model  # Ensure Model enum is defined

# Configure logging
logging.basicConfig(
    filename='./classification_log.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features(feature_generator, dataloader, device='cpu'):
    features = []
    labels = []
    feature_generator.model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            output = feature_generator.generate(images)
            output = output.cpu().numpy()
            output = output.reshape(output.shape[0], -1)
            features.append(output)
            labels.append(targets.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def main():
    # Set random seed
    set_seed()

    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    logging.info(f"Using device: {DEVICE}")

    # Define normalization transforms
    normalize_imagenet = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]    
    )
    
    normalize_endo = transforms.Normalize(
        mean=[0.5929, 0.3667, 0.1843],
        std=[0.1932, 0.1411, 0.094]
    )
    
    # Define transformations for different model groups
    transform_imagenet = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        normalize_imagenet, 
    ])
    
    transform_endo = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        normalize_endo, 
    ])
    
    # Load dataset paths
    dataset_path = os.path.expanduser('~/TUD/InformatikUni/B_Thesis/data/cedex')
    train_dir = os.path.join(dataset_path, 'Train')
    test_dir = os.path.join(dataset_path, 'Test')
    
    # Verify that the directories exist
    if not os.path.isdir(train_dir):
        logging.error(f"Training directory not found at {train_dir}")
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.isdir(test_dir):
        logging.error(f"Testing directory not found at {test_dir}")
        raise FileNotFoundError(f"Testing directory not found at {test_dir}")
    
    # Load datasets with different transforms
    train_dataset_imagenet = datasets.ImageFolder(root=train_dir, transform=transform_imagenet)
    test_dataset_imagenet = datasets.ImageFolder(root=test_dir, transform=transform_imagenet)
    
    train_dataset_endo = datasets.ImageFolder(root=train_dir, transform=transform_endo)
    test_dataset_endo = datasets.ImageFolder(root=test_dir, transform=transform_endo)
    
    # Create DataLoaders
    batch_size = 64
    num_workers = 0 if DEVICE == 'mps' else 4
    
    train_loader_imagenet = DataLoader(
        train_dataset_imagenet, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE != 'cpu' else False
    )
    test_loader_imagenet = DataLoader(
        test_dataset_imagenet, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if DEVICE != 'cpu' else False
    )
    
    train_loader_endo = DataLoader(
        train_dataset_endo, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if DEVICE != 'cpu' else False
    )
    test_loader_endo = DataLoader(
        test_dataset_endo, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if DEVICE != 'cpu' else False
    )
    
    # Initialize models using FeatureGenerator
    feature_generators = {
        'CendoFM': FeatureGenerator(Model.CENDO_FM),
        'EndoFM': FeatureGenerator(Model.ENDO_FM),
        'Swin': FeatureGenerator(Model.Swin_v2_B),
        'ResNet50': FeatureGenerator(Model.RES_NET_50),
    }
    
    # Log model initialization
    for name in feature_generators.keys():
        logging.info(f"Initialized FeatureGenerator for {name}")
    
    # Define model groups based on required normalization
    endo_models = ['CendoFM', 'EndoFM']
    imagenet_models = ['Swin', 'ResNet50']
    
    # Initialize dictionaries to store features and labels
    train_features = {}
    train_labels = {}
    test_features = {}
    test_labels = {}
    
    # Extract features for EndoFM-normalized models
    for name in endo_models:
        fg = feature_generators[name]
        logging.info(f"Extracting features for {name} on training data with EndoFM normalization...")
        print(f"Extracting features for {name} on training data with EndoFM normalization...")
        features, labels = extract_features(fg, train_loader_endo, DEVICE)
        train_features[name] = features
        train_labels[name] = labels
    
    for name in endo_models:
        fg = feature_generators[name]
        logging.info(f"Extracting features for {name} on test data with EndoFM normalization...")
        print(f"Extracting features for {name} on test data with EndoFM normalization...")
        features, labels = extract_features(fg, test_loader_endo, DEVICE)
        test_features[name] = features
        test_labels[name] = labels
    
    # Extract features for ImageNet-normalized models
    for name in imagenet_models:
        fg = feature_generators[name]
        logging.info(f"Extracting features for {name} on training data with ImageNet normalization...")
        print(f"Extracting features for {name} on training data with ImageNet normalization...")
        features, labels = extract_features(fg, train_loader_imagenet, DEVICE)
        train_features[name] = features
        train_labels[name] = labels
    
    for name in imagenet_models:
        fg = feature_generators[name]
        logging.info(f"Extracting features for {name} on test data with ImageNet normalization...")
        print(f"Extracting features for {name} on test data with ImageNet normalization...")
        features, labels = extract_features(fg, test_loader_imagenet, DEVICE)
        test_features[name] = features
        test_labels[name] = labels
    
    # Define the range of k values to test
    k_values = list(range(5,50,5)) 
    
    # Initialize a dictionary to store results
    results = {name: {'k': [], 'accuracy': []} for name in feature_generators.keys()}
    
    # Train and evaluate k-NN classifiers for each k
    for name in tqdm(feature_generators.keys()):
        logging.info(f"Evaluating k-NN classifiers for {name}...")
        print(f"\nEvaluating k-NN classifiers for {name}...")
        for k in k_values:
            logging.info(f"Testing k={k} for {name}...")
            print(f"  Testing k={k}...")
            knn = KNeighborsClassifier(n_neighbors=k, weights="distance") 
            knn.fit(train_features[name], train_labels[name])
            predictions = knn.predict(test_features[name])
            acc = accuracy_score(test_labels[name], predictions)
            results[name]['k'].append(k)
            results[name]['accuracy'].append(acc)
            logging.info(f"k={k}, Accuracy={acc:.4f} for {name}")
            print(f"    k={k}, Accuracy: {acc:.4f}")
    
    # Convert results to a DataFrame for better visualization
    df_results = pd.DataFrame(columns=['Model', 'k', 'Accuracy'])
    for name in results.keys():
        for k, acc in zip(results[name]['k'], results[name]['accuracy']):
            df_results = df_results.append({'Model': name, 'k': k, 'Accuracy': acc}, ignore_index=True)
    
    # Display the pivoted DataFrame
    pivot_df = df_results.pivot(index='k', columns='Model', values='Accuracy')
    print("\nSummary of k-NN Classifier Performance:")
    print(pivot_df)
    logging.info("Summary of k-NN Classifier Performance:")
    logging.info("\n" + pivot_df.to_string())
    
    # Save the results to a CSV file
    os.makedirs('results', exist_ok=True)
    df_results.to_csv('results/knn_k_values_results.csv', index=False)
    print("Results have been saved to 'results/knn_k_values_results.csv'.")
    logging.info("Results have been saved to 'results/knn_k_values_results.csv'.")
    
    # Plotting the results
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    for name in feature_generators.keys():
        subset = df_results[df_results['Model'] == name]
        sns.lineplot(x='k', y='Accuracy', data=subset, label=name, marker='o')
    
    plt.title('k-NN Classifier Accuracy for Different k Values')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.legend(title='Model')
    plt.xticks(k_values)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/knn_k_values_plot.png')
    print("Plot has been saved to 'results/knn_k_values_plot.png'.")
    logging.info("Plot has been saved to 'results/knn_k_values_plot.png'.")
    
    # Display the plot
    plt.show()
    
    # Identify the best k for each model
    best_k = df_results.loc[df_results.groupby('Model')['Accuracy'].idxmax()]
    print("\nOptimal k for each model based on highest accuracy:")
    print(best_k)
    logging.info("Optimal k for each model based on highest accuracy:")
    logging.info(best_k.to_string())
    
    # Save the best k to a CSV file
    best_k.to_csv('results/knn_best_k_values.csv', index=False)
    print("Best k values have been saved to 'results/knn_best_k_values.csv'.")
    logging.info("Best k values have been saved to 'results/knn_best_k_values.csv'.")

if __name__ == "__main__":
    main()