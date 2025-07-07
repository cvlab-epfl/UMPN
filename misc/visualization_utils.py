import cv2
import numpy as np
from typing import List
from misc.log_utils import log

def save_images_as_video(images: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Save a list of images as an MP4 video.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays (BGR format).
        output_path (str): Path to save the output video file.
        fps (int): Frames per second for the output video. Default is 30.

    Returns:
        None
    """
    if not images:
        log.error("The list of images is empty.")
        raise ValueError("The list of images is empty.")

    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, image in enumerate(images):
        if image.shape[:2] != (height, width):
            log.error(f"Image {i} has different dimensions than the first image.")
            raise ValueError(f"Image {i} has different dimensions than the first image.")
        
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] != 3:  # Not BGR
            log.error(f"Image {i} is not in BGR, RGBA, or grayscale format.")
            raise ValueError(f"Image {i} is not in BGR, RGBA, or grayscale format.")

        out.write(image)

    out.release()
    log.info(f"Video saved successfully at {output_path}")

