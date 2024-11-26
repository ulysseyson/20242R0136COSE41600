import cv2
import os

from tqdm import tqdm

# Directory containing the PNG files
p = "02_straight_duck_walk"
image_folder = f"tmp/data/{p}/pcd"
output_file = f"{p}.mp4"

# Get all PNG files in the directory, sorted by filename
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # Ensure the images are in order

# Read the first image to determine the frame size
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, _ = frame.shape

# Define video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for .avi files
fps = 30  # Frames per second
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Loop through images and write them to the video
for image in tqdm(images):
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    out.write(frame)

# Release the VideoWriter object
out.release()

print(f"Video saved as {output_file}")
