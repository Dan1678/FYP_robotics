import sys
import os
import socket
import time
import torch
import matplotlib.pyplot as plt
import json


from SingleRobotSystem.single_robot_system import plan_and_execute, load_robot_coord_mapping, generate_task_details
from Planning.gpt_functions import extract_task_objects, generate_open_verification_prompt, chat_with_gpt
from Perception.segmentation_layer import perform_segmentation, encode_and_match
from Execution.client_script import send_command_to_robot
from Mapping import image_to_robo_mapping


'''
This file contains the Dual-Robot system

'''

# --- Configuration ---
PRIMARY_IP     = 'xxxx'   # Worker robot IP
PRIMARY_PORT   = 'xxxx'
VISION_IP      = 'xxxx'  # Camera robot IP
VISION_PORT    = 'xxxx'

with open("bin_calibration_simple.json", "r") as f:
    _bin_map = json.load(f)

GREEN_PIXEL = tuple(_bin_map["Green Bin"]["pixel"])
GREEN_POSE6 = tuple(_bin_map["Green Bin"]["pose6d"])
BLUE_PIXEL  = tuple(_bin_map["Blue Bin"]["pixel"])
BLUE_POSE6  = tuple(_bin_map["Blue Bin"]["pose6d"])

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

TABLE_THRESH = 0.1
BIN_THRESH   = 0.1

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
        resp = sock.recv(1024).decode().strip()
        if resp != "DONE":
            print(f"[Verifier] Unexpected response: {resp}")
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
        best_idxs, scores = encode_and_match(
            cropped_images, task_objects, device, return_scores=True
        )
        for i, obj in enumerate(task_objects):
            conf = scores[i] if i < len(scores) else 0.0
            table_confidences[obj] = float(conf)
            if conf > 0 and i < len(best_idxs):
                idx = best_idxs[i]
                if 0 <= idx < len(centers):
                    pixel = centers[idx]
                    pose6d = image_to_robo_mapping.find_closest_gripper_point(pixel, mapping)
                    table_poses[obj] = tuple(pose6d)
                else:
                    table_poses[obj] = None
            else:
                table_poses[obj] = None
    else:
        for obj in task_objects:
            table_confidences[obj] = 0.0
            table_poses[obj] = None

    print(f"[Verifier] Table confidences: {table_confidences}")
    return table_confidences, table_poses

def verify_bin_scene(task_objects, device):
    """
    Capture bin-view, segment & CLIP-match for each task object,
    then assign each object to the closest bin reference pixel.

    Returns:
      - bin_confidences: dict object_name -> confidence
      - bin_poses:       dict object_name -> 6D pose or None
    """
    print("\n[Verifier] Capturing bin-view for verification…")
    original, cropped = perform_segmentation()
    bin_confidences = {}
    bin_poses = {}

    if cropped:
        cropped_images, centers = zip(*cropped)
        _, scores = encode_and_match(cropped_images, task_objects, device, return_scores=True)

        for i, obj in enumerate(task_objects):
            conf = scores[i] if i < len(scores) else 0.0
            bin_confidences[obj] = float(conf)

            if conf >= BIN_THRESH:
                u_m, v_m = centers[i]

                # Compare squared pixel‐space distances:
                u_g, v_g = GREEN_PIXEL
                u_b, v_b = BLUE_PIXEL
                d2g = (u_m - u_g)**2 + (v_m - v_g)**2
                d2b = (u_m - u_b)**2 + (v_m - v_b)**2

                if d2g <= d2b:
                    bin_poses[obj] = GREEN_POSE6
                else:
                    bin_poses[obj] = BLUE_POSE6
            else:
                bin_poses[obj] = None
    else:
        for obj in task_objects:
            bin_confidences[obj] = 0.0
            bin_poses[obj] = None

    print(f"[Verifier] Bin confidences: {bin_confidences}")
    return bin_confidences, bin_poses


def main():
    # 1) Ensure camera starts over the table
    print("\n[Verifier] Moving camera to table-view (home)…")
    send_vision_command("home")

    # 2) Load gripper-coordinate mapping once
    mapping = load_robot_coord_mapping()
    if not mapping:
        print("Failed to load robot coord mapping. Exiting.")
        return

    # 3) Read and parse the user's task
    task_desc = input("Enter your task description: ")
    task_objs = extract_task_objects(task_desc)
    task_objs = [o for o in task_objs if "bin" not in o.strip().lower()]
    print(f"Task objects: {task_objs}")
    if not task_objs:
        print("No objects to handle. Exiting.")
        return

    # Determine which bin was requested in the task
    desc_lower = task_desc.lower()
    if "green bin" in desc_lower:
        desired_bin = "Green Bin"
    elif "blue bin" in desc_lower:
        desired_bin = "Blue Bin"
    else:
        desired_bin = "Green Bin"  # default fallback

    device = get_device()

    # 4) Execute the worker-robot plan (single pass)
    success = plan_and_execute(task_desc, task_objs, mapping)
    if not success:
        print("\n Worker failed to complete the plan. Exiting.")
        return

    # 5) End-of-task table-view verification
    print("\n[Verifier] Moving camera to table-view (home)…")
    send_vision_command("home")
    table_confidences, table_poses = verify_table_scene(task_objs, device, mapping)

    # 6) End-of-task bin-view verification
    print("\n[Verifier] Moving camera to bin-view…")
    send_vision_command("bins")
    bin_confidences, bin_poses = verify_bin_scene(task_objs, device)

    # 7) Build the GPT verification prompt
    prompt = generate_open_verification_prompt(
        task_description=task_desc,
        table_confidences=table_confidences,
        table_poses=table_poses,
        bin_confidences=bin_confidences,
        bin_poses=bin_poses,
        bin_fixed_poses=BIN_FIXED_POSES
    )
    print("\n[Verifier] GPT Verification Prompt:\n")
    print(prompt)

    verification_reply = chat_with_gpt(prompt)
    print("\n[Verifier] GPT Verification Reply:\n")
    print(verification_reply)

    # 8) If GPT says "Yes", we’re done
    reply_lower = verification_reply.strip().lower()
    if reply_lower.startswith("yes"):
        print("\n🎉 All objects verified as correctly placed. Task complete.")
        return

    print("\n[Verifier] GPT indicates objects remain misplaced. Performing one retry…")

    # 9) Decide which objects to retry:
    to_retry = []
    for obj in task_objs:
        t_conf = table_confidences.get(obj, 0.0)
        b_conf = bin_confidences.get(obj, 0.0)
        if b_conf <= t_conf or b_conf < BIN_THRESH:
            to_retry.append(obj)

    if not to_retry:
        print("\n[Verifier] No clear misplaced objects to retry. Exiting.")
        return

    # 10) For each misplaced object, generate a mini pick-and-place to the desired bin
    for obj in to_retry:
        pose6d = table_poses.get(obj)
        if not pose6d:
            print(f"[Verifier] Cannot retry '{obj}'—no valid table pose. Skipping.")
            continue

        pick_x, pick_y, pick_z, pick_r, pick_p, pick_yaw = pose6d
        bin_data = BIN_FIXED_POSES[desired_bin]
        place_x, place_y, place_z = bin_data["position"]
        place_r, place_p, place_yaw = bin_data["orientation"]

        # Build a minimal objects_dict for retry planning
        retry_objects = {
            obj: {
                "position": (pick_x, pick_y, pick_z),
                "orientation": (pick_r, pick_p, pick_yaw)
            },
            desired_bin: {
                "position": (place_x, place_y, place_z),
                "orientation": (place_r, place_p, place_yaw)
            }
        }

        retry_prompt = (
            f"Pick up the {obj} from "
            f"({pick_x:.2f}, {pick_y:.2f}, {pick_z:.2f}; "
            f"{pick_r:.2f}, {pick_p:.2f}, {pick_yaw:.2f}) "
            f"and place it into the {desired_bin} "
            f"({place_x:.2f}, {place_y:.2f}, {place_z:.2f}; "
            f"{place_r:.2f}, {place_p:.2f}, {place_yaw:.2f})."
        )
        print(f"\n[Verifier] Retry GPT Prompt for {obj}:\n{retry_prompt}")

        details = generate_task_details(retry_prompt, retry_objects)
        instrs = __import__('client_script').generate_instructions(details)

        # Move vision robot out of the way before sending retry commands
        print("[Verifier] Moving vision robot to home (out of way)…")
        send_vision_command("home")
        time.sleep(1)

        print(f"[Verifier] Sending retry commands for '{obj}'…")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((PRIMARY_IP, PRIMARY_PORT))
            for cmd in instrs:
                send_command_to_robot(sock, cmd)
                time.sleep(1)

    # After the single retry, we stop here. User will verify by eye.
    print("\n[Verifier] Retry commands dispatched. Please verify placement visually.")
    return

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"\nElapsed time: {end - start:.4f} seconds")
