import ffmpeg
import argparse
import os

# Set up argument parser to get the input directory
parser = argparse.ArgumentParser("Mp4 to jpgs")
parser.add_argument("d", help="directory of videos")
parser.add_argument("f",help="fps", default=30)
args = parser.parse_args()

# Get the list of all MP4 files in the provided directory
video_dir = args.d  
mp4_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
fps = args.f

# Process each MP4 file in the directory
for video_file in mp4_files:
    input_path = os.path.join(video_dir, video_file)
    patient_id = os.path.splitext(video_file)[0]  # Remove the .mp4 extension

    # Create an output directory based on the video file name
    output_dir = os.path.join(video_dir, patient_id,patient_id)
    
    # Check if the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use ffmpeg to extract frames and save them inside the created directory
    (ffmpeg
     .input(input_path)
     .filter('fps', fps=fps, round='up')  # Set frame extraction rate
     .output(f"{output_dir}/{patient_id}_%05d.jpg", **{'qscale:v': 3})  # Save images inside the directory
     .run())

    print(f"Extracted frames for {video_file} into {output_dir}")