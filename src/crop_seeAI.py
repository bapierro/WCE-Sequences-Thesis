import os
import math
from PIL import Image
from tqdm import tqdm

def preprocess_dataset(
    source_dir,
    dest_dir,
    circle_radius=284,
    file_extensions=('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')
):
    """
    Preprocess images by performing a center crop of the largest square
    that fits within a circle of the given radius and saving the processed images
    to a new directory.

    Args:
        source_dir (str): Path to the original dataset directory.
        dest_dir (str): Path to the destination directory for processed images.
        circle_radius (int): Radius of the circle to determine the size of the center crop.
        file_extensions (tuple): Image file extensions to process.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Calculate the side length of the largest square that fits in the circle
    crop_side = int(circle_radius * math.sqrt(2))
    crop_size = (crop_side, crop_side)

    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Compute the corresponding destination directory
        relative_path = os.path.relpath(root, source_dir)
        dest_subdir = os.path.join(dest_dir, relative_path)
        os.makedirs(dest_subdir, exist_ok=True)

        # Process each image file
        for filename in tqdm(files, desc=f"Processing images in {root}"):
            if filename.lower().endswith(file_extensions):
                source_path = os.path.join(root, filename)
                dest_path = os.path.join(dest_subdir, filename)

                # Open the image
                try:
                    with Image.open(source_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Perform center crop
                        img_width, img_height = img.size
                        crop_width, crop_height = crop_size

                        # Ensure crop size does not exceed image dimensions
                        crop_width = min(crop_width, img_width)
                        crop_height = min(crop_height, img_height)

                        left = (img_width - crop_width) // 2
                        top = (img_height - crop_height) // 2
                        right = left + crop_width
                        bottom = top + crop_height
                        img_cropped = img.crop((left, top, right, bottom))

                        # Save the processed image
                        img_cropped.save(dest_path)
                except Exception as e:
                    print(f"Error processing {source_path}: {e}")
            else:
                # Optionally handle non-image files
                pass  # Or remove this else block if not required

if __name__ == "__main__":
    source_dirs = [
        "data/Kid_no_classes",
    ]
    dest_dirs = [
        "data/kid_processsed_training",
    ]

    for source_dir, dest_dir in zip(source_dirs, dest_dirs):
        print(f"Processing dataset: {source_dir}")
        preprocess_dataset(
            source_dir=source_dir,
            dest_dir=dest_dir,
            circle_radius=174
        )
