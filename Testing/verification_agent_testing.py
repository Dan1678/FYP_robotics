import sys
import os
import socket
import time
import torch
import matplotlib.pyplot as plt
import time

from SingleRobotSystem.single_robot_system import load_robot_coord_mapping
from Planning.gpt_functions import extract_task_objects, generate_open_verification_prompt, chat_with_gpt
from Perception.segmentation_layer import perform_segmentation, encode_and_match
from Mapping.image_to_robo_mapping import find_closest_gripper_point

VISION_IP      = 'XXXX'  # Camera robot IP
VISION_PORT    = 'XXXX'
TABLE_THRESH   = 0.1
BIN_THRESH     = 0.1

# Fixed bin poses (6D)
BIN_FIXED_POSES = {
    "Green Bin": {
        "position": (172.8, -226.4, 107.4),
        "orientation": (93.9, -0.83, 47.41)
    },
    "Blue Bin": {
        "position": (166.7, -273.3, 108.2),
        "orientation": (73.14, -1.83, 12.7)
    }
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def send_vision_command(cmd: str):
    """
    Send 'home' or 'bins' to the vision robot and wait for 'DONE'.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((VISION_IP, VISION_PORT))
        sock.sendall(f"{cmd}\n".encode())
        response = sock.recv(1024).decode().strip()
        if response != "DONE":
            print(f"[Verifier] Unexpected response: {response}")
        time.sleep(1)
def verify_table_scene(task_objects, device, mapping):
    """
    Capture table-view, segment, CLIP-match, and return confidences + 6D poses.
    """
    print("\n[Verifier] Capturing table-view for verification…")
    _, cropped = perform_segmentation()
    table_confidences = {}
    table_poses = {}

    if cropped:
        cropped_images, centers = zip(*cropped)
        # Get both match indices and confidence scores
        best_idxs, scores = encode_and_match(
            cropped_images, task_objects, device, return_scores=True
        )
        for i, obj in enumerate(task_objects):
            # If we have a score for this object, use it; otherwise assume 0.0
            conf = scores[i] if i < len(scores) else 0.0
            table_confidences[obj] = float(conf)

            # Only map to a 6D pose if confidence is nonzero and best_idxs[i] is valid
            if conf > 0 and i < len(best_idxs):
                idx = best_idxs[i]
                if 0 <= idx < len(centers):
                    pixel = centers[idx]
                    pose6d = find_closest_gripper_point(pixel, mapping)
                    table_poses[obj] = tuple(pose6d)
                else:
                    table_poses[obj] = None
            else:
                table_poses[obj] = None
    else:
        # No masks found → all confidences zero, all poses None
        for obj in task_objects:
            table_confidences[obj] = 0.0
            table_poses[obj] = None

    print(f"[Verifier] Table confidences: {table_confidences}")
    return table_confidences, table_poses


def verify_bin_scene(task_objects, device):
    """
    Capture bin-view, segment, CLIP-match, and return confidences + 6D poses.
    """
    print("\n[Verifier] Capturing bin-view for verification…")
    _, cropped = perform_segmentation()
    bin_confidences = {}
    bin_poses = {}

    if cropped:
        cropped_images, _ = zip(*cropped)
        # Get both match indices and confidence scores
        best_idxs, scores = encode_and_match(
            cropped_images, task_objects, device, return_scores=True
        )
        for i, obj in enumerate(task_objects):
            conf = scores[i] if i < len(scores) else 0.0
            bin_confidences[obj] = float(conf)

            # If confidence is nonzero and index is valid, assign Green Bin pose
            if conf > 0 and i < len(best_idxs):
                g = BIN_FIXED_POSES["Green Bin"]
                bx, by, bz = g["position"]
                br, bp, byw = g["orientation"]
                bin_poses[obj] = (bx, by, bz, br, bp, byw)
            else:
                bin_poses[obj] = None
    else:
        for obj in task_objects:
            bin_confidences[obj] = 0.0
            bin_poses[obj] = None

    print(f"[Verifier] Bin confidences: {bin_confidences}")
    return bin_confidences, bin_poses

def main():

    # 1) Move camera to table-view
    print("\n[Verifier] Moving camera to table-view (home)…")
    send_vision_command("home")

    # 2) Load mapping
    mapping = load_robot_coord_mapping()
    if not mapping:
        print("Failed to load robot coord mapping. Exiting.")
        return

    # 3) Read task and extract objects
    task_desc = input("Enter your task description: ")
    task_objs = extract_task_objects(task_desc)
    task_objs = [o for o in task_objs if "bin" not in o.strip().lower()]
    print(f"Task objects: {task_objs}")
    if not task_objs:
        print("No objects to handle. Exiting.")
        return

    device = get_device()

    # 4) Table-view verification
    print("\n[Verifier] Moving camera to table-view (home)…")
    send_vision_command("home")
    table_confidences, table_poses = verify_table_scene(task_objs, device, mapping)

    # 5) Bin-view verification
    print("\n[Verifier] Moving camera to bin-view…")
    send_vision_command("bins")
    bin_confidences, bin_poses = verify_bin_scene(task_objs, device)

    # 6) Build and send verification prompt (including bin_poses)
    prompt = generate_open_verification_prompt(
        task_description=task_desc,
        table_confidences=table_confidences,
        table_poses=table_poses,
        bin_confidences=bin_confidences,
        bin_fixed_poses=BIN_FIXED_POSES,
        bin_poses=bin_poses
    )
    print("\n[Verifier] GPT Verification Prompt:\n")
    print(prompt)

    verification_reply = chat_with_gpt(prompt)
    print("\n[Verifier] GPT Verification Reply:\n")
    print(verification_reply)

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.4f} seconds")
