

import os
import sys
import argparse
import cairosvg

def convert_svg_to_png(input_path, output_path=None):
    """
    Converts an SVG file to a PNG file using CairoSVG.
    
    Parameters:
        input_path (str): Full path to the input SVG file.
        output_path (str): Full path to the output PNG file. If None, a .png 
                           with the same name is created in the same directory.
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".png"
    cairosvg.svg2png(url=input_path, write_to=output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert all SVG files in a directory to PNG.")
    parser.add_argument("directory", nargs="?", help="The directory containing SVG files")
    args = parser.parse_args()

    if not args.directory:
        # If no directory is provided, prompt the user
        args.directory = input("Please enter the directory containing SVG files: ").strip()

    input_dir = args.directory

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    # Find all SVG files in the directory
    svg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".svg")]

    if not svg_files:
        print("No SVG files found in the specified directory.")
        sys.exit(0)

    # Create output directory for PNGs if desired
    # (Uncomment and modify if you want a separate output directory)
    # output_dir = os.path.join(input_dir, "pngs")
    # os.makedirs(output_dir, exist_ok=True)

    for svg_file in svg_files:
        input_path = os.path.join(input_dir, svg_file)
        # If using an output directory:
        # output_path = os.path.join(output_dir, os.path.splitext(svg_file)[0] + ".png")
        # Otherwise, just store PNG alongside the SVG:
        output_path = None
        print(f"Converting {svg_file}...")
        convert_svg_to_png(input_path, output_path)

    print("Conversion complete.")

if __name__ == "__main__":
    main()
