import sys
import os
import socket
import time
import torch
import matplotlib.pyplot as plt


from Planning.gpt_functions import extract_task_features, extract_task_objects, generate_task_details
from Perception.segmentation_layer import perform_segmentation, encode_and_match
from Execution.client_script import generate_instructions, send_command_to_robot
from Mapping.image_to_robo_mapping import load_robot_coord_mapping, find_closest_gripper_point

# --- Configuration ---
PRIMARY_IP   = 'XXXX'
PRIMARY_PORT = 'XXXX'

# --- Helpers ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def plan_and_execute(task_description: str, task_objects: list, mapping: dict) -> bool:
    """
    Segment the scene, match task_objects, generate instructions via GPT,
    and send them to the primary robot. Returns True on success.
    """
    print("\n[Primary] Capturing scene and segmenting…")
    original, cropped = perform_segmentation()
    if not cropped:
        print("[Primary] No objects segmented.")
        return False

    cropped_images, centers = zip(*cropped)
    device = get_device()
    best_idxs = encode_and_match(cropped_images, task_objects, device)
    if not best_idxs or len(best_idxs) != len(task_objects):
        print("[Primary] CLIP matching failed.")
        return False

    objects_dict = {}
    for obj, idx in zip(task_objects, best_idxs):
        img_center = centers[idx]
        gripper = find_closest_gripper_point(img_center, mapping)
        objects_dict[obj] = {
            "position": tuple(gripper[:3]),
            "orientation": tuple(gripper[3:6])
        }
        # visualize for verification
        plt.figure(figsize=(4, 4))
        plt.imshow(cropped_images[idx])
        plt.title(f"{obj} @ {img_center}")
        plt.axis("off")
        plt.show()

    # add bin (fixed location for placing objects)
    objects_dict["Green Bin"] = {
        "position": (172.8, -226.4, 107.4),
        "orientation": (93.9, -0.83, 47.41)
    }
    objects_dict["Blue Bin"] = {
        "position": (166.7, -273.3, 108.2),
        "orientation": (73.14, -1.83, 12.7)     
    }

    # build task details and generate instructions
    details = generate_task_details(task_description, objects_dict)
    print("\n[Primary] Task details:\n", details)
    instrs = generate_instructions(details)
    print(f"\n[Primary] Sending {len(instrs)} commands…")

    # send to primary robot
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((PRIMARY_IP, PRIMARY_PORT))
        for cmd in instrs:
            send_command_to_robot(sock, cmd)
            time.sleep(1)

    return True

def test_1_robot():
    """
    Single-robot test function: runs mapping update once, extracts task info,
    and executes the plan without any verification loops.
    """
    # if not update_point_mapping():
    #     print("Mapping update failed. Exiting.")
    #     return False

    # Load the gripper-coordinate mapping only once
    mapping = load_robot_coord_mapping()
    if not mapping:
        print("Failed to load robot coord mapping. Exiting.")
        return False

    # 1) Task description, features, objects
    task_desc = input("Enter your task description: ")
    features = extract_task_features(task_desc)
    print("Extracted features:", features)

    task_objs = extract_task_objects(task_desc)
    # Remove any mention of 'bin' from the object list
    task_objs = extract_task_objects(task_desc)
    # Remove any object whose name contains 'bin' (e.g. "Green Bin", "Blue Bin")
    task_objs = [o for o in task_objs if "bin" not in o.strip().lower()]

    if not task_objs:
        print("No objects to handle. Exiting.")
        return False

    # 2) Single execution (no verification loop), passing preloaded mapping
    success = plan_and_execute(task_desc, task_objs, mapping)
    if success:
        print("\n Task executed successfully!")
    else:
        print("\n Task execution failed.")

    return success

if __name__ == "__main__":
    test_1_robot()
