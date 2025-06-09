#!/usr/bin/env python3
import sys
import os
import socket
import cv2
import numpy as np
from scipy.spatial import KDTree


from Mapping.image_to_robo_mapping import load_robot_coord_mapping, find_closest_gripper_point
from Execution.client_script import send_command_to_robot

CAMERA_INDEX = "XXXX"                    # OpenCV camera index (e.g., 0 or 1)
HOST         = "XXXX"      # Robot 1 IP address (example)
PORT         = "XXXX"                # Robot 1 TCP port

# ─── Mouse callback storage ────────────────────────────────────────────────────
clicked_points = []

def on_mouse(event, x, y, flags, param):
    """
    Mouse callback to record clicked points.
    Left-click to select a point; the (x, y) pixel coordinate is stored.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point selected: (x={x}, y={y})")
        # Draw a small circle at the clicked location
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibration Image", param)

def main():
    # 1) Capture a single frame from the camera for point selection
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        return

    # Display the captured frame and allow the user to click points
    display_img = frame.copy()
    cv2.namedWindow("Calibration Image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Calibration Image", on_mouse, display_img)
    print("Click points on the image to test mapping. Press 'q' to finish.")
    while True:
        cv2.imshow("Calibration Image", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

    if not clicked_points:
        print("No points were selected. Exiting.")
        return

    # 2) Load the KD-tree fallback mapping
    point_mapping = load_robot_coord_mapping()  # dict: {(x_img, y_img): [6D gripper pose]}
    img_pts = np.array(list(point_mapping.keys()), dtype=np.float32)
    kd_tree = KDTree(img_pts)

    # 3) Open TCP connection to Robot 1
    try:
        sock = socket.create_connection((HOST, PORT), timeout=5)
    except Exception as e:
        print(f"Error: Could not connect to {HOST}:{PORT} -> {e}")
        return

    # 4) For each clicked point, map to nearest 6D pose, send command,
    #    then wait for user to input measured distance and record it.
    measured_errors = []

    for idx, (x_img, y_img) in enumerate(clicked_points, start=1):
        # Use KD-tree to find nearest mapped image point
        _, nearest_idx = kd_tree.query([x_img, y_img])
        nearest_key = tuple(img_pts[nearest_idx])
        gripper_pose = point_mapping[nearest_key]  # [X, Y, Z, pitch, roll, yaw]

        # Construct move command string
        x, y, z, pitch, roll, yaw = gripper_pose
        cmd = f"move({x:.2f},{y:.2f},{z:.2f},{pitch:.2f},{roll:.2f},{yaw:.2f})"
        print(f"\n[{idx}/{len(clicked_points)}] Sending command: {cmd}")

        # Send the command
        try:
            send_command_to_robot(sock, cmd)
        except Exception as e:
            print(f"  ERROR: Failed to send '{cmd}': {e}")

        # Wait for user to measure and enter the actual distance error
        while True:
            user_input = input(f"Enter measured distance error for pose {idx} (in same units as robot coords): ")
            try:
                error_val = float(user_input)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value (e.g., 12.34).")

        measured_errors.append(error_val)
        print(f"Recorded error for pose {idx}: {error_val}\n")

    sock.close()
    print("All commands sent and measurements recorded. Computing RMSE...")

    # 5) Compute RMSE
    errors_array = np.array(measured_errors, dtype=np.float64)
    rmse = np.sqrt(np.mean(errors_array ** 2))
    print(f"RMSE over {len(measured_errors)} points: {rmse:.4f}")

if __name__ == "__main__":
    main()
