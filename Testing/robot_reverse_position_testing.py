#!/usr/bin/env python3
import sys
import os
import socket
import time
import json
from Execution.client_script import send_command_to_robot

# ─── Configuration ─────────────────────────────────────────────────────────────
HOST          = "XXXX"     # Robot 1 IP address
PORT          = "XXXX"               # Robot 1 TCP port
DELAY_SECONDS = 5.0                # Seconds to wait between moves
MAPPING_JSON  = os.path.join("point_mapping.json")


def main():
    # 1) Load the 2D → 6D mapping table
    if not os.path.isfile(MAPPING_JSON):
        print(f"Error: point_mapping.json not found at {MAPPING_JSON}")
        return

    with open(MAPPING_JSON, "r") as f:
        mapping = json.load(f)

    # 2) Extract keys in insertion order and reverse them
    #    JSON loading in Python preserves insertion order for dicts.
    all_keys = list(mapping.keys())
    reversed_keys = list(reversed(all_keys))

    # 3) Connect to Robot 1 via TCP
    try:
        sock = socket.create_connection((HOST, PORT), timeout=5)
    except Exception as e:
        print(f"Error: Could not connect to {HOST}:{PORT} → {e}")
        return

    # 4) Iterate from the 12th point down to the 1st point
    total = len(reversed_keys)
    for idx, key in enumerate(reversed_keys, start=1):
        pose6d = mapping[key]  # [X, Y, Z, pitch, roll, yaw]
        x, y, z, pitch, roll, yaw = pose6d
        cmd = f"move({x:.2f},{y:.2f},{z:.2f},{pitch:.2f},{roll:.2f},{yaw:.2f})"

        print(f"[{idx}/{total}] Key = {key} → 6D = {pose6d}")
        print(f"Sending command: {cmd}")
        try:
            send_command_to_robot(sock, cmd)
        except Exception as e:
            print(f"Error sending command for key {key}: {e}")
            break

        print(f"Waiting for {DELAY_SECONDS} seconds before next move...")
        time.sleep(DELAY_SECONDS)

    sock.close()
    print("Finished sending all reversed-point commands.")


if __name__ == "__main__":
    main()
