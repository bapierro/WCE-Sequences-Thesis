import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd

# Import the FeatureGenerator and Model classes
from feature_generator import FeatureGenerator  # Ensure this path is correct
from model_name import Model  # Ensure this path is correct

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Rotational Invariance of a Model")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the image dataset')
    parser.add_argument('--model_name', type=str, required=True, choices=[model.name for model in Model], help='Name of the model to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for transformations')
    parser.add_argument('--output_dir', type=str, default='./rotation_evaluation', help='Directory to save evaluation results')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights for the model')
    parser.add_argument('--student', action='store_true', help='Use student model features if applicable')
    parser.add_argument('--cb', type=str, default=None, help='Checkpoint path for CENDO_FM model if applicable')
    args = parser.parse_args()
    return args

def get_rotations():
    """
    Returns a list of rotation angles.
    """
    return [0, 90, 180, 270]

def apply_rotations(image, angles):
    """
    Applies specified rotations to an image.

    Parameters:
    - image: PIL Image
    - angles: List of angles to rotate

    Returns:
    - List of rotated PIL Images
    """
    rotated_images = []
    for angle in angles:
        rotated = transforms.functional.rotate(image, angle)
        rotated_images.append(rotated)
    return rotated_images

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of PIL Images and labels.

    Parameters:
    - batch: List of tuples (PIL.Image.Image, label)

    Returns:
    - images: List of PIL.Image.Image
    - labels: List of labels
    """
    images, labels = zip(*batch)
    return list(images), list(labels)

def compute_similarity(original_features, rotated_features):
    """
    Computes similarity metrics between original and rotated features.

    Parameters:
    - original_features: NumPy array of shape (batch_size, feature_dim)
    - rotated_features: NumPy array of shape (batch_size, num_rotations, feature_dim)

    Returns:
    - similarities_cosine: NumPy array of shape (batch_size, num_rotations)
    - similarities_dot: NumPy array of shape (batch_size, num_rotations)
    - similarities_euclidean: NumPy array of shape (batch_size, num_rotations)
    """
    # Compute Cosine Similarity
    # Reshape for sklearn's cosine_similarity
    batch_size, num_rotations, feature_dim = rotated_features.shape
    original_features_reshaped = original_features.reshape(batch_size, 1, feature_dim)
    rotated_features_reshaped = rotated_features.reshape(batch_size * num_rotations, feature_dim)

    # Cosine Similarity
    cosine_sim = cosine_similarity(original_features_reshaped.reshape(-1, feature_dim), rotated_features_reshaped)
    cosine_sim = cosine_sim.reshape(batch_size, num_rotations)

    # Dot Product Similarity
    dot_sim = np.einsum('ijk,ik->ij', rotated_features, original_features)

    # Euclidean Distance
    # Compute Euclidean distances between each original and rotated feature
    euclidean_dist = euclidean_distances(original_features, rotated_features.reshape(batch_size * num_rotations, feature_dim))
    euclidean_dist = euclidean_dist.reshape(batch_size, num_rotations)

    return cosine_sim, dot_sim, euclidean_dist

def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize FeatureGenerator
    try:
        model_enum = Model[args.model_name]
    except KeyError:
        raise ValueError(f"Model name '{args.model_name}' is not valid. Choose from {[model.name for model in Model]}")

    feature_generator = FeatureGenerator(
        model_name=model_enum,
        pretrained=args.pretrained,
        img_size=args.img_size,
        student=args.student,
        cb=args.cb
    ).to(DEVICE)

    feature_generator.model.eval()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Adjust if different
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset without any transformation
    dataset = datasets.ImageFolder(root=args.dataset_path, transform=None)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn  # Use the custom collate function
    )

    # Initialize similarity records
    rotation_angles = get_rotations()
    num_rotations = len(rotation_angles) - 1  # Exclude 0 degrees as it's the original
    similarity_records_cosine = {angle: [] for angle in rotation_angles if angle != 0}
    similarity_records_dot = {angle: [] for angle in rotation_angles if angle != 0}
    similarity_records_euclidean = {angle: [] for angle in rotation_angles if angle != 0}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Processing Batches")):
            # 'images' is a list of PIL Images
            pil_images = images  # Direct assignment for clarity

            # Apply rotations
            all_rotated_images = []
            for img in pil_images:
                rotated = apply_rotations(img, rotation_angles)
                all_rotated_images.extend(rotated)  # List of rotated PIL Images

            # Convert all rotated images to tensors
            all_rotated_tensors = [transform(img) for img in all_rotated_images]
            all_rotated_tensors = torch.stack(all_rotated_tensors).to(DEVICE)  # Shape: (batch_size * num_rotations, C, H, W)

            # Generate features
            features = feature_generator.generate(all_rotated_tensors)  # Shape: (batch_size * num_rotations, feature_dim)

            # Reshape to (batch_size, num_rotations, feature_dim)
            batch_size = len(pil_images)
            rotated_features = features.view(batch_size, len(rotation_angles), -1)  # Shape: (batch_size, 4, feature_dim)

            # Separate original and rotated features
            original_features = rotated_features[:, 0, :].cpu().numpy()  # Shape: (batch_size, feature_dim)
            rotated_features_np = rotated_features[:, 1:, :].cpu().numpy()  # Shape: (batch_size, 3, feature_dim)

            # Compute similarities
            similarities_cosine, similarities_dot, similarities_euclidean = compute_similarity(original_features, rotated_features_np)
            # Each shape: (batch_size, 3)

            # Record similarities
            for i, angle in enumerate(rotation_angles[1:]):
                similarity_records_cosine[angle].extend(similarities_cosine[:, i])
                similarity_records_dot[angle].extend(similarities_dot[:, i])
                similarity_records_euclidean[angle].extend(similarities_euclidean[:, i])

    # Aggregate and report results
    report_path = os.path.join(args.output_dir, "rotational_invariance_report.csv")
    report_data = []

    for angle in sorted(similarity_records_cosine.keys()):
        sim_cosine = np.array(similarity_records_cosine[angle])
        sim_dot = np.array(similarity_records_dot[angle])
        sim_euclidean = np.array(similarity_records_euclidean[angle])

        report_data.append({
            'Rotation Angle (degrees)': angle,
            'Mean Cosine Similarity': np.mean(sim_cosine),
            'Median Cosine Similarity': np.median(sim_cosine),
            'Std Dev Cosine Similarity': np.std(sim_cosine),
            'Min Cosine Similarity': np.min(sim_cosine),
            'Max Cosine Similarity': np.max(sim_cosine),
            'Mean Dot Product Similarity': np.mean(sim_dot),
            'Median Dot Product Similarity': np.median(sim_dot),
            'Std Dev Dot Product Similarity': np.std(sim_dot),
            'Min Dot Product Similarity': np.min(sim_dot),
            'Max Dot Product Similarity': np.max(sim_dot),
            'Mean Euclidean Distance': np.mean(sim_euclidean),
            'Median Euclidean Distance': np.median(sim_euclidean),
            'Std Dev Euclidean Distance': np.std(sim_euclidean),
            'Min Euclidean Distance': np.min(sim_euclidean),
            'Max Euclidean Distance': np.max(sim_euclidean)
        })

        print(f"Rotation {angle}°:")
        print(f"  Cosine Similarity - Mean: {np.mean(sim_cosine):.4f}, Median: {np.median(sim_cosine):.4f}, Std Dev: {np.std(sim_cosine):.4f}, Min: {np.min(sim_cosine):.4f}, Max: {np.max(sim_cosine):.4f}")
        print(f"  Dot Product Similarity - Mean: {np.mean(sim_dot):.4f}, Median: {np.median(sim_dot):.4f}, Std Dev: {np.std(sim_dot):.4f}, Min: {np.min(sim_dot):.4f}, Max: {np.max(sim_dot):.4f}")
        print(f"  Euclidean Distance - Mean: {np.mean(sim_euclidean):.4f}, Median: {np.median(sim_euclidean):.4f}, Std Dev: {np.std(sim_euclidean):.4f}, Min: {np.min(sim_euclidean):.4f}, Max: {np.max(sim_euclidean):.4f}\n")

    # Save report to CSV
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(report_path, index=False)
    print(f"Rotational invariance report saved to {report_path}")

    # Optional: Plot similarity distributions

    # Cosine Similarity
    plt.figure(figsize=(12, 8))
    for angle in sorted(similarity_records_cosine.keys()):
        sns.kdeplot(similarity_records_cosine[angle], label=f"{angle}° Rotation")
    plt.title("Cosine Similarity Distributions for Rotated Images")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plot_path_cosine = os.path.join(args.output_dir, "cosine_similarity_distributions.png")
    plt.savefig(plot_path_cosine, bbox_inches='tight')
    print(f"Cosine similarity distribution plot saved to {plot_path_cosine}")
    plt.close()

    # Dot Product Similarity
    plt.figure(figsize=(12, 8))
    for angle in sorted(similarity_records_dot.keys()):
        sns.kdeplot(similarity_records_dot[angle], label=f"{angle}° Rotation")
    plt.title("Dot Product Similarity Distributions for Rotated Images")
    plt.xlabel("Dot Product Similarity")
    plt.ylabel("Density")
    plt.legend()
    plot_path_dot = os.path.join(args.output_dir, "dot_product_similarity_distributions.png")
    plt.savefig(plot_path_dot, bbox_inches='tight')
    print(f"Dot product similarity distribution plot saved to {plot_path_dot}")
    plt.close()

    # Euclidean Distance
    plt.figure(figsize=(12, 8))
    for angle in sorted(similarity_records_euclidean.keys()):
        sns.kdeplot(similarity_records_euclidean[angle], label=f"{angle}° Rotation")
    plt.title("Euclidean Distance Distributions for Rotated Images")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Density")
    plt.legend()
    plot_path_euclidean = os.path.join(args.output_dir, "euclidean_distance_distributions.png")
    plt.savefig(plot_path_euclidean, bbox_inches='tight')
    print(f"Euclidean distance distribution plot saved to {plot_path_euclidean}")
    plt.close()