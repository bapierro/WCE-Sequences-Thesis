import os
import shutil
from PIL import Image, ImageDraw
from tqdm import tqdm

def preprocess_dataset(
    source_dir,
    dest_dir,
    remove_box=True,
    circle_radius=288,
    crop_size=(512, 512),
    file_extensions=('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')
):
    """
    Preprocess images by painting black outside a central circle,
    performing a center crop, and saving the processed images to a new directory.

    Args:
        source_dir (str): Path to the original dataset directory.
        dest_dir (str): Path to the destination directory for processed images.
        remove_box (bool): Whether to apply the circular mask.
        circle_radius (int): Radius of the central circle to keep.
        crop_size (tuple): Size of the center crop (width, height).
        file_extensions (tuple): Image file extensions to process.
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

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

                        # Optionally apply the circular mask
                        if remove_box:
                            # Create a mask with a white circle on black background
                            mask = Image.new('L', img.size, 0)
                            draw = ImageDraw.Draw(mask)
                            center = (img.width // 2, img.height // 2)
                            left_up_point = (center[0] - circle_radius, center[1] - circle_radius)
                            right_down_point = (center[0] + circle_radius, center[1] + circle_radius)
                            draw.ellipse([left_up_point, right_down_point], fill=255)

                            # Create a black image for the background
                            black_bg = Image.new('RGB', img.size, (0, 0, 0))

                            # Composite the original image with the black background using the mask
                            img = Image.composite(img, black_bg, mask)

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
        "data/see_ai_ordered",
    ]
    dest_dirs = [
        "data/see_ai_processed",
    ]

    for source_dir, dest_dir in zip(source_dirs, dest_dirs):
        print(f"Processing dataset: {source_dir}")
        preprocess_dataset(
            source_dir=source_dir,
            dest_dir=dest_dir,
            remove_box=True,
            circle_radius=288,
            crop_size=(512, 512)
        )