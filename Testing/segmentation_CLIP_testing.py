#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import torch
import time

from Perception.segmentation_layer import perform_segmentation
from Perception.clip_layer        import encode_and_match

def test_segmentation_and_clip(task_objects):
    """
    1) Runs SAM2 segmentation via perform_segmentation().
    2) Draws bounding boxes & centers on the original image.
    3) Uses CLIP to find the best‐matching mask for each task object.
    4) Displays the overlay and top crops via OpenCV windows.
    5) Returns a dict mapping each object label to its chosen mask index.
    """
    # 1) Segment + get crops with center coords
    pil_image, cropped_with_centers = perform_segmentation()
    image = np.array(pil_image)  # RGB uint8

    # 2) Overlay boxes & centers
    vis = image.copy()
    for crop, (cx, cy) in cropped_with_centers:
        h, w = crop.shape[:2]
        x0, y0 = int(cx - w/2), int(cy - h/2)
        x1, y1 = x0 + w, y0 + h
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    cv2.imshow("Segmentation Masks Overlay", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    # 3) Prepare crops for CLIP
    crops = [crop for crop, _ in cropped_with_centers]

    # 4) Match with CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_indices = encode_and_match(crops, task_objects, device)

    # 5) Display best matches
    results = {}
    for obj, idx in zip(task_objects, best_indices):
        results[obj] = idx
        print(f"Best match for '{obj}': mask #{idx}")
        top_crop = crops[idx]
        cv2.imshow(f"Top Crop for '{obj}'", cv2.cvtColor(top_crop, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return results

if __name__ == "__main__":
    # Example: test with a single object
    t_start = time.perf_counter()
    labels = ["carrot"]
    matches = test_segmentation_and_clip(labels)
    print("Final matches:", matches)
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"... in {elapsed:.3f} s")

