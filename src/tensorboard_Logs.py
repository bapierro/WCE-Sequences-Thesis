import os
import tensorflow as tf

def list_tensorboard_tags(log_dir):
    """
    List all scalar tags in TensorBoard event files.
    
    Parameters:
    - log_dir (str): Path to the TensorBoard log directory.
    """
    tags = set()
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(root, file)
                for event in tf.compat.v1.train.summary_iterator(event_file):
                    for value in event.summary.value:
                        tags.add(value.tag)
    return tags

if __name__ == "__main__":
    log_dir = "tb_logs"  # Replace with your actual log directory
    tags = list_tensorboard_tags(log_dir)
    print("Available TensorBoard Tags:")
    for tag in sorted(tags):
        print(tag)