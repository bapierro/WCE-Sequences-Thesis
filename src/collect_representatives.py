import os
import glob
import math
import base64
import svgwrite
from model_name import Model  # Ensure that Model is correctly imported

def collect_images(root_dir):
    """
    Traverse the directory structure to collect all .png image paths.

    :param root_dir: The root directory to start searching from.
    :return: List of image file paths.
    """
    print(f"Collecting images from: {root_dir}")
    # Use glob to recursively find all png files in Cluster_X directories
    pattern = os.path.join(root_dir, '**', 'Cluster_*', '*.png')
    image_paths = glob.glob(pattern, recursive=True)
    print(f"Found {len(image_paths)} images.")
    return image_paths

def encode_image_to_base64(image_path):
    """
    Encode an image to base64 to embed in SVG.

    :param image_path: Path to the image file.
    :return: Base64 encoded string of the image.
    """
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def calculate_grid_dimensions(num_images):
    """
    Calculate the number of rows and columns for the grid to make it as square as possible.

    :param num_images: Total number of images.
    :return: Tuple of (rows, cols)
    """
    if num_images == 0:
        return 0, 0
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    return rows, cols

def create_svg_grid(image_paths, output_svg_path, image_size=100, spacing=10):
    """
    Create an SVG file with images arranged in a grid.

    :param image_paths: List of image file paths.
    :param output_svg_path: Path to save the generated SVG.
    :param image_size: Size (width and height) of each image in pixels.
    :param spacing: Space between images in pixels.
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("No images found to display.")
        return

    rows, cols = calculate_grid_dimensions(num_images)
    print(f"Arranging {num_images} images into a grid of {rows} rows and {cols} columns.")

    # Calculate SVG canvas size
    canvas_width = cols * image_size + (cols + 1) * spacing
    canvas_height = rows * image_size + (rows + 1) * spacing

    # Initialize SVG drawing
    dwg = svgwrite.Drawing(output_svg_path, size=(canvas_width, canvas_height))
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))  # Optional: white background

    for idx, image_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        x = spacing + col * (image_size + spacing)
        y = spacing + row * (image_size + spacing)

        # Encode image to base64
        img_data = encode_image_to_base64(image_path)
        img_extension = os.path.splitext(image_path)[1].lower()
        if img_extension in ['.jpg', '.jpeg']:
            img_format = 'jpeg'
        elif img_extension == '.png':
            img_format = 'png'
        else:
            print(f"Unsupported image format for file {image_path}. Skipping.")
            continue

        # Create image element
        img = dwg.image(href=f"data:image/{img_format};base64,{img_data}", insert=(x, y), size=(image_size, image_size))
        dwg.add(img)

    # Save the SVG
    dwg.save()
    print(f"SVG grid saved to {output_svg_path}")

def main(models, output_directory="svg_grids", image_size=100, spacing=10):
    """
    Generate SVG grids for each model.

    :param models: List of Model enums.
    :param output_directory: Directory where SVGs will be saved.
    :param image_size: Size of each image in the grid.
    :param spacing: Spacing between images in the grid.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for model in models:
        model_name = model.name  # Assuming Model is an Enum with a 'name' attribute
        root_dir = os.path.join(
            "dumps",
            "representatives",
            "20241110_153354",
        )

        # Verify that the root directory exists
        if not os.path.isdir(root_dir):
            print(f"Directory does not exist: {root_dir}. Skipping model: {model_name}")
            continue

        # Collect all image paths
        image_paths = collect_images(root_dir)

        if not image_paths:
            print(f"No images found for model {model_name}. Skipping.")
            continue

        # Define the output SVG file path with model name
        output_svg_filename = f"cluster_grid_{model_name}.svg"
        output_svg_path = os.path.join(output_directory, output_svg_filename)

        # Create the SVG grid
        create_svg_grid(image_paths, output_svg_path, image_size, spacing)

if __name__ == "__main__":
    # Define the models you want to generate grids for
    models = [Model.CENDO_FM, Model.ENDO_FM, Model.RES_NET_101]

    # Optional: Define output directory, image size, and spacing
    output_directory = "svg_grids"  # Directory to save all SVG grids
    image_size = 100  # pixels
    spacing = 0       # pixels

    main(models=models, output_directory=output_directory, image_size=image_size, spacing=spacing)