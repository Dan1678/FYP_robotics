# bin_mapping_simple.py

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def capture_raw_frame(camera_index=1):
    """
    Capture a single RGB frame from the specified camera index.
    Returns a NumPy array of shape (H, W, 3), in RGB order.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

def on_click(event, click_list):
    """
    Matplotlib callback: record (u, v) on left‐click.
    """
    if event.inaxes and event.button == 1:  # left‐click only
        u, v = event.xdata, event.ydata
        click_list.append((u, v))
        plt.scatter([u], [v], c="r", s=50)
        plt.draw()

def collect_one_bin(bin_name, img_np):
    """
    Display `img_np` with Matplotlib, let user click exactly one point, then prompt for its 6D pose.
    Returns:
      - uv: (u, v) pixel coordinate
      - pose6d: (X, Y, Z, roll, pitch, yaw)
    """
    print(f"\nClick exactly one point on the {bin_name}.")
    clicked = []

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_np)
    ax.set_title(f"Click one point on the {bin_name}")
    cid = fig.canvas.mpl_connect(
        "button_press_event", lambda ev: on_click(ev, clicked)
    )

    # Wait until the user has clicked once
    while len(clicked) < 1:
        plt.pause(0.1)

    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    u, v = clicked[0]
    print(f"  → You clicked pixel (u={u:.1f}, v={v:.1f}) on the {bin_name}.")

    # Prompt for 6D pose
    print(f"Enter the 6D robot pose at that point for the {bin_name}:")
    print("  Format: X Y Z roll pitch yaw  (six floats, space‐separated)")
    while True:
        entry = input(">> ").strip().split()
        if len(entry) == 6:
            try:
                X, Y, Z, r, p, yaw = map(float, entry)
                break
            except ValueError:
                pass
        print("Invalid. Please enter exactly six floats (e.g. `172.8 -226.4 107.4 93.9 -0.83 47.41`).")

    return (u, v), (X, Y, Z, r, p, yaw)

def main():
    # 1) Capture a single raw frame from camera index 1
    print("Capturing one frame from camera…")
    img_np = capture_raw_frame(camera_index=1)

    # 2) Let user click each bin once and enter its 6D pose
    mapping = {}
    for bin_name in ["Green Bin", "Blue Bin"]:
        uv, pose6d = collect_one_bin(bin_name, img_np)
        mapping[bin_name] = {
            "pixel": [float(uv[0]), float(uv[1])],
            "pose6d": [pose6d[0], pose6d[1], pose6d[2],
                       pose6d[3], pose6d[4], pose6d[5]]
        }

    # 3) Save to JSON
    out_path = "bin_calibration_simple.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nWrote simple bin calibration to '{out_path}'.\n"
          "You can now use this JSON in verify_bin_scene(...) to decide nearest bin.")

if __name__ == "__main__":
    main()
