import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Define the dense sections
dense_sections = [
    np.arange(0, 200, 0.5),       # Very high density
    np.arange(200, 5000, 2),      # High density
    np.arange(5000, 10000, 10),   # Moderate density
    np.arange(10000, 20000, 50),  # Low density
    np.arange(20000, 50000, 100), # Sparse density
    np.arange(50000, 80000, 500), # Very sparse density
    np.arange(80000, 100000, 1000) # Ultra-sparse density
]

# Combine all sections into one time array
time_variable_density = np.concatenate(dense_sections)

# Generate noisy data corresponding to time
np.random.seed(2)  # For reproducibility
data_variable_density = np.random.normal(0, 10, size=len(time_variable_density))

# Introduce jumps at random points
jump_directions = np.random.choice([-1, 1], size=10)  # Random up or down
jump_positions = np.random.choice(len(time_variable_density), size=10, replace=False)

for i, pos in enumerate(jump_positions):
    data_variable_density[pos:] += jump_directions[i] * np.random.uniform(20, 40)

# Apply Gaussian smoothing to observe the effect on trends
smoothed_variable_density = gaussian_filter1d(data_variable_density, sigma=100)

# Prepare 2D data for HDBSCAN
data_for_clustering = np.vstack((time_variable_density, smoothed_variable_density)).T

# ------------------- Apply HDBSCAN with Soft Clustering -------------------
# Set prediction_data=True to enable soft clustering features
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=1, prediction_data=True)
clusterer.fit(data_for_clustering)

# Get soft cluster membership probabilities
# Each row corresponds to a point, and each column corresponds to a cluster
membership_probs = hdbscan.all_points_membership_vectors(clusterer)

# Assign each point to the cluster with the highest membership probability
max_probs = np.max(membership_probs, axis=1)
labels = np.argmax(membership_probs, axis=1)

# Assign noise label (-1) to points where the maximum probability is zero (i.e., noise points)
labels[max_probs == 0] = -1

# Extract unique labels after soft clustering
unique_labels = np.unique(labels)

# ------------------- Create Side-by-Side Plots -------------------
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# --- Plot 1: Original Sections ---
colors = plt.get_cmap('tab10', len(dense_sections))
start_idx = 0

for i, section in enumerate(dense_sections):
    end_idx = start_idx + len(section)
    axs[0].scatter(
        time_variable_density[start_idx:end_idx],
        smoothed_variable_density[start_idx:end_idx],
        color=colors(i),
        s=10,
        label=f"Section {i+1}"
    )
    start_idx = end_idx

axs[0].set_title("Original Time Series Sections")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Smoothed Value")
axs[0].legend()

# --- Plot 2: HDBSCAN Soft Clustering ---
for label in unique_labels:
    if label == -1:  # Noise points
        axs[1].scatter(
            time_variable_density[labels == label],
            smoothed_variable_density[labels == label],
            color="gray",
            s=10,
            label="Noise"
        )
    else:
        axs[1].scatter(
            time_variable_density[labels == label],
            smoothed_variable_density[labels == label],
            s=10,
            label=f"Cluster {label}"
        )

axs[1].set_title("HDBSCAN Soft Clustering Results")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Smoothed Value")
axs[1].legend()

plt.tight_layout()
plt.show()