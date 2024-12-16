import os
import numpy as np
import torch
import argparse
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm


FLAGS = argparse.ArgumentParser("Dataset Report")

FLAGS.add_argument("-d",help="Dataset path")

def calculate_statistics(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    sum_of_squares = torch.zeros(3)
    n_samples = 0
    pixel_hist = [np.zeros(256) for _ in range(3)]
    correlation_matrix = torch.zeros(3, 3)
    entropies = []

    for images, _ in tqdm(dataloader,"Calculating statistics",unit="batch"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)

        # Mean and sum of squares for std calculation
        mean += images.mean(2).sum(0)
        sum_of_squares += images.pow(2).mean(2).sum(0)
        n_samples += batch_samples

        # Calculate histograms
        for i in range(3):  # RGB channels
            channel_data = images[:, i, :].flatten().cpu().numpy()
            hist, _ = np.histogram(channel_data, bins=256, range=(0, 1))
            pixel_hist[i] += hist

        # Calculate correlation between channels
        for i in range(3):
            for j in range(3):
                if i != j:
                    correlation_matrix[i, j] += torch.sum((images[:, i, :] - mean[i]) * (images[:, j, :] - mean[j]))

        # Calculate image entropy
        for img in images:
            img_entropy = 0
            for i in range(3):  # RGB channels
                channel_hist, _ = np.histogram(img[i].cpu().numpy(), bins=256, range=(0, 1))
                channel_entropy = entropy(channel_hist + 1e-9, base=2)  # Add small value to avoid log(0)
                img_entropy += channel_entropy
            entropies.append(img_entropy / 3.0)

    mean /= n_samples
    std = torch.sqrt(sum_of_squares / n_samples - mean ** 2)
    correlation_matrix /= n_samples * images.size(2)

    return mean, std, pixel_hist, correlation_matrix, np.mean(entropies)


def save_report(mean, std, correlation_matrix, avg_entropy, report_path,dataset_name):
    with open(report_path, 'w') as report_file:
        report_file.write(f"{dataset_name} Statistics Report\n")
        report_file.write("===============================\n\n")
        report_file.write(f"Mean (RGB): {mean}\n")
        report_file.write(f"Std (RGB): {std}\n")
        report_file.write("\nChannel Correlations:\n")
        report_file.write(f"{correlation_matrix}\n")
        report_file.write(f"\nAverage Entropy: {avg_entropy:.4f}\n")


def save_histograms(pixel_hist, output_dir):
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.figure()
        plt.plot(pixel_hist[i], color=color)
        plt.title(f'Pixel Value Histogram - {color.capitalize()} Channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{color}_histogram.png'))
        plt.close()


def main():
    args = FLAGS.parse_args()
    path = args.d
    dataset_name = str(path).split(os.path.sep)[-1]
    transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
    ])

    # Load your dataset
    dataset = datasets.ImageFolder(root=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Create stats directory
    output_dir = f"./stats/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    mean, std, pixel_hist, correlation_matrix, avg_entropy = calculate_statistics(dataloader)

    # Save report
    report_path = os.path.join(output_dir, f"{dataset_name}_report.txt")
    save_report(mean, std, correlation_matrix, avg_entropy, report_path,dataset_name)

    # Save histograms
    save_histograms(pixel_hist, output_dir)

    print(f"Report and histograms saved in directory: {output_dir}")


if __name__ == '__main__':
    main()