import cv2
import os

from tqdm import tqdm

# Directory containing the PNG files
ps = [
    "01_straight_walk",
    "02_straight_duck_walk",
    "03_straight_crawl",
    "04_zigzag_walk",
    "05_straight_duck_walk",
    "06_straight_crawl",
    "07_straight_walk",
]

for p in ps:
    image_folder = f"tmp/try4/{p}"
    output_file = f"tmp/try4/{p}.mp4"

    # Get all PNG files in the directory, sorted by filename
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the images are in order

    # Read the first image to determine the frame size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for .avi files
    fps = 15  # Frames per second
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Loop through images and write them to the video
    for image in tqdm(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f"Video saved as {output_file}")
