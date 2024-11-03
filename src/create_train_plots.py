import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.lines import Line2D  # For custom legend handles

# Suppress TensorFlow deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
warnings.filterwarnings('ignore', category=DeprecationWarning)

def extract_scalars_event_accumulator(event_file, tags):
    """
    Extract scalar values using TensorBoard's EventAccumulator.

    Parameters:
    - event_file (str): Path to the TensorBoard event file.
    - tags (list): List of tags to extract.

    Returns:
    - data (dict): Dictionary with tags as keys and lists of (step, value) tuples.
    """
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    data = {}
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            scalars = ea.Scalars(tag)
            data[tag] = [(s.step, s.value) for s in scalars]
    return data

def main(log_dir, output_dir):
    """
    Main function to traverse log directories, extract scalars, and plot combined loss curves.

    Parameters:
    - log_dir (str): Path to the TensorBoard log directory.
    - output_dir (str): Directory to save the plots.
    """
    # Define the tags to extract
    tags_to_extract = ['train_loss_step', 'train_loss_epoch', 'val_loss', 'epoch']

    # Get all experiment directories at the specified log directory level
    experiments = sorted([d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))])
    
    # Set Seaborn palette to 'tab10'
    sns.set_palette("tab10")

    if not experiments:
        print(f"No experiment directories found under '{log_dir}'.")
        return

    # Initialize empty DataFrames for training and validation
    train_data_list = []
    val_data_list = []

    for experiment_name in experiments:
        experiment_path = os.path.join(log_dir, experiment_name)
        versions = [d for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

        if not versions:
            print(f"No version directories found under '{experiment_path}'. Skipping.")
            continue

        for version in versions:
            version_path = os.path.join(experiment_path, version)
            event_files = [
                os.path.join(version_path, f)
                for f in os.listdir(version_path)
                if f.startswith("events.out.tfevents")
            ]

            if not event_files:
                print(f"No event files found in '{version_path}'. Skipping.")
                continue

            for event_file in event_files:
                # Extract the data
                run_data = extract_scalars_event_accumulator(event_file, tags_to_extract)

                # Prepare DataFrames
                df_train_loss_step = pd.DataFrame(run_data.get('train_loss_step', []), columns=['step', 'train_loss'])
                df_epoch = pd.DataFrame(run_data.get('epoch', []), columns=['step', 'epoch'])
                df_val_loss = pd.DataFrame(run_data.get('val_loss', []), columns=['step', 'val_loss'])

                if df_train_loss_step.empty or df_epoch.empty:
                    print(f"Insufficient data in '{version_path}'. Skipping.")
                    continue

                # Map steps to epochs for training loss
                df_train_loss_step.sort_values('step', inplace=True)
                df_epoch.sort_values('step', inplace=True)
                df_train_loss_with_epoch = pd.merge_asof(
                    df_train_loss_step, df_epoch, on='step', direction='backward'
                )

                # Compute mean training loss per epoch
                df_mean_train_loss_per_epoch = (
                    df_train_loss_with_epoch.groupby('epoch')['train_loss'].mean().reset_index()
                )
                df_mean_train_loss_per_epoch['experiment'] = experiment_name

                train_data_list.append(df_mean_train_loss_per_epoch)

                # Map steps to epochs for validation loss
                if not df_val_loss.empty:
                    df_val_loss.sort_values('step', inplace=True)
                    df_val_loss_with_epoch = pd.merge_asof(
                        df_val_loss, df_epoch, on='step', direction='backward'
                    )
                    # Compute mean validation loss per epoch
                    df_mean_val_loss_per_epoch = (
                        df_val_loss_with_epoch.groupby('epoch')['val_loss'].mean().reset_index()
                    )
                    df_mean_val_loss_per_epoch['experiment'] = experiment_name

                    val_data_list.append(df_mean_val_loss_per_epoch)
                else:
                    print(f"No validation loss data for '{experiment_name}' in version '{version}'.")

    # Concatenate all collected data
    train_df = pd.concat(train_data_list, ignore_index=True) if train_data_list else pd.DataFrame()
    val_df = pd.concat(val_data_list, ignore_index=True) if val_data_list else pd.DataFrame()

    # Create a single plot for both Training and Validation Loss
    fig, ax = plt.subplots(figsize=(15, 8))

    # Get the color palette used by Seaborn
    palette = sns.color_palette("tab10")
    experiment_colors = {experiment: palette[i % len(palette)] for i, experiment in enumerate(experiments)}

    # Plot Training Loss with transparency
    if not train_df.empty:
        sns.lineplot(
            data=train_df,
            x='epoch',
            y='train_loss',
            hue='experiment',
            palette=palette,
            ax=ax,
            alpha=0.3,
            linewidth=2,
        )
    else:
        print("No training data to plot.")

    # Plot Validation Loss with solid lines
    if not val_df.empty:
        sns.lineplot(
            data=val_df,
            x='epoch',
            y='val_loss',
            hue='experiment',
            palette=palette,
            ax=ax,
            alpha=1.0,
            linewidth=2,
        )

        # Annotate minimum validation loss for each experiment
        for experiment in val_df['experiment'].unique():
            exp_val_df = val_df[val_df['experiment'] == experiment]
            if not exp_val_df.empty:
                min_val_loss = exp_val_df['val_loss'].min()
                min_epoch = exp_val_df.loc[exp_val_df['val_loss'].idxmin(), 'epoch']
                
                # Get the color assigned to the experiment from the palette
                color = experiment_colors[experiment]

                # Plot the minimum point
                # ax.scatter(min_epoch, min_val_loss, color=color, marker='d', s=100, zorder=5)

                # # Annotate the minimum point
                # ax.text(
                #     min_epoch, min_val_loss, f'{min_val_loss:.4f}',
                #     color=color,
                #     fontsize=9,
                #     va='bottom',
                #     ha='center',
                #     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                # )
    else:
        print("No validation data to plot.")

    # Set titles and labels
    ax.set_title('Training and Validation Loss Curves', fontsize=18)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(True, which='both', linestyle='-', alpha=0.7)
    sns.despine(trim=True, offset=10)

    # Create custom legend handles for experiments (solid lines)
    experiment_handles = []
    for experiment in experiments:
        color = experiment_colors[experiment]
        handle = Line2D([0], [0], color=color, linewidth=2, label=experiment)
        experiment_handles.append(handle)

    # Create custom legend handles for metrics
    metrics_handles = [
        Line2D([0], [0], color='black', linewidth=2, alpha=1.0, label='Validation Loss'),
        Line2D([0], [0], color='black', linewidth=2, alpha=0.3, label='Training Loss')
    ]

    # Add the legends to the plot
    legend1 = ax.legend(
        handles=experiment_handles,
        title='Setting',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=metrics_handles,
        title='Phase',
        bbox_to_anchor=(1.05, 0.7),
        loc='upper left',
        borderaxespad=0.
    )

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    plot_filename = "training_and_validation_loss_curves.png"
    plot_path = os.path.join(output_dir, plot_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Training and validation loss curves saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot Combined Training and Validation Loss Over Epochs from TensorBoard Logs'
    )
    parser.add_argument(
        '--logs', type=str, required=True,
        help='Path to TensorBoard log directory (e.g., tb_logs)'
    )
    parser.add_argument(
        '--output', type=str, default='plots',
        help='Directory to save the plots'
    )
    args = parser.parse_args()

    log_dir = args.logs
    output_dir = args.output

    main(log_dir, output_dir)