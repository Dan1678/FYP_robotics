import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
from clip_layer import encode_and_match
from Mapping.image_to_robo_mapping import load_robot_coord_mapping, find_closest_gripper_point  # Import functions

# The file is uses the SAM2 model to segment images.

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Function to display masks
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) 
                        for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
    ax.imshow(img)

def perform_segmentation():
    print("Capturing image from camera...")
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        return None, []

    # convert to RGB
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_np)
    H, W = image_np.shape[:2]

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    print("Building SAM model...")
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    print("Initializing mask generator...")
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    print("Generating masks...")
    masks = mask_generator.generate(image_np)
    print(f"Number of masks generated: {len(masks)}")
    if masks:
        print(f"Keys in first mask: {masks[0].keys()}")

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

    # --- Filter, crop and zero-background ---
    cropped_images_with_centers = []
    for mask in masks:
        seg = mask['segmentation'].astype(bool)
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        # skip overly large masks
        if w * h > 0.8 * H * W:
            continue
        # crop and zero out background
        crop = image_np[y0:y0+h, x0:x0+w].copy()
        crop[~seg[y0:y0+h, x0:x0+w]] = 255
        cx, cy = x0 + w // 2, y0 + h // 2
        cropped_images_with_centers.append((crop, (cx, cy)))

    # If no valid masks after filtering, return empty list instead of raising
    if not cropped_images_with_centers:
        print("No valid masks found (table appears empty).")
        return image, []

    # optional: show bounding boxes on original
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    ax = plt.gca()
    for crop, (cx, cy) in cropped_images_with_centers:
        # draw small circle at center
        ax.plot(cx, cy, 'yo', markersize=10)
    plt.axis('off')
    plt.show()

    return image, cropped_images_with_centers

