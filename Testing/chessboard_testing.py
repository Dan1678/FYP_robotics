#!/usr/bin/env python3
import sys
import os
import socket
import time
import numpy as np
from scipy.spatial import KDTree


# ─── Imports ───────────────────────────────────────────────────────────────────
from Mapping.image_to_robo_mapping import load_robot_coord_mapping
from Execution.client_script import send_command_to_robot

HOST         = "XXXX"   # <— replace with your Robot 1’s IP
PORT         = "XXXX"             # <— replace with the correct TCP port
DELAY        = 10              # seconds between commands (increase if too fast)

def main():
    # 1) Load the 12‐point image→6D mapping dictionary
    #    load_robot_coord_mapping() returns a dict whose keys are (x_img, y_img)
    #    and whose values are [X, Y, Z, pitch, roll, yaw].
    try:
        point_map = load_robot_coord_mapping()
    except Exception as e:
        print(f"Error: could not load point mapping → {e}")
        return

    # Convert keys to a numpy array of shape (12, 2) for KDTree indexing.
    img_pts = np.array(list(point_map.keys()), dtype=np.float32)
    # Build a KD‐tree over those 2D image points
    kd_tree = KDTree(img_pts)

    # 2) Open a TCP connection to Robot 1
    try:
        sock = socket.create_connection((HOST, PORT), timeout=5)
    except Exception as e:
        print(f"Error: Could not connect to {HOST}:{PORT} → {e}")
        return

    # 3) Iterate over each of the 12 stored image keys in sorted order
    #    (Sorting ensures a reproducible order.)
    sorted_keys = sorted(point_map.keys(), key=lambda tup: (tup[1], tup[0]))
    print("Will send move() commands for these 12 image→6D entries (sorted by y, then x):")
    for idx, key in enumerate(sorted_keys, start=1):
        x_img, y_img = key
        gripper_pose = point_map[key]  # [X, Y, Z, pitch, roll, yaw]

        # Build the "move(...)" string
        X, Y, Z, pitch, roll, yaw = gripper_pose
        cmd = f"move({X:.2f},{Y:.2f},{Z:.2f},{pitch:.2f},{roll:.2f},{yaw:.2f})"

        print(f"[{idx}/12] Image‐pixel = ({x_img:.1f}, {y_img:.1f}) → 6D = {gripper_pose}")
        print(f"        Sending command: {cmd}")
        try:
            send_command_to_robot(sock, cmd)
        except Exception as e:
            print(f"        Error sending command: {e}")

        # Wait a bit so you can physically observe each motion
        time.sleep(DELAY)

    sock.close()
    print("All 12 commands sent. Done.")

if __name__ == "__main__":
    main()
