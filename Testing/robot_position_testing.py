#!/usr/bin/env python3
import sys
import os
import socket
import time
from Execution.client_script import send_command_to_robot

HOST      =  "XXXX"  # Replace with your Robot 1’s IP
PORT      =  "XXXX"             # Replace with your Robot 1’s TCP port
DELAY     = 1.5              # Seconds to wait between each move (increase if needed)
TIMEOUT   = 5                # seconds for socket timeout

fixed_poses = [
    [
        84.2,
        -301.6,
        91.5,
        66.96,
        0.99,
        -1.27
    ],
    [
        39.4,
        -303.6,
        91.5,
        68.6,
        -3.89,
        -1.7
    ],
    [
        1.4,
        -305.7,
        92.3,
        70.39,
        -7.33,
        -5.28
    ],
    [
        -37.0,
        -304.1,
        90.2,
        72.21,
        -7.12,
        -3.07
    ],
    [
        85.1,
        -290.6,
        106.2,
        83.59,
        0.29,
        1.74
    ],
    [
        43.3,
        -293.3,
        105.7,
        84.22,
        -0.84,
        -0.11
    ],
    [
        11.7,
        -294.2,
        100.5,
        85.92,
        -1.68,
        -0.53
    ],
    [
        -27.2,
        -294.6,
        102.7,
        86.64,
        -1.58,
        0.76
    ],
    [
        74.1,
        -256.3,
        109.7,
        78.6,
        -5.85,
        -1.7
    ],
    [
        35.9,
        -255.6,
        103.6,
        81.74,
        -3.38,
        -0.58
    ],
    [
        2.2,
        -257.8,
        106.9,
        82.66,
        -0.93,
        0.17
    ],
    [
        -37.5,
        -253.3,
        109.3,
        81.03,
        -0.45,
        1.89
    ]
]


def format_move_command(pose):
    """
    Given a 6-element list [X, Y, Z, pitch, roll, yaw],
    return a string: move(X,Y,Z,pitch,roll,yaw) with two-decimal formatting.
    """
    x, y, z, pitch, roll, yaw = pose
    return f"move({x:.2f},{y:.2f},{z:.2f},{pitch:.2f},{roll:.2f},{yaw:.2f})"


def main():
    print("Opening TCP connection to robot at {}:{} ...".format(HOST, PORT))
    try:
        sock = socket.create_connection((HOST, PORT), timeout=TIMEOUT)
    except Exception as e:
        print(f"ERROR: Unable to connect to {HOST}:{PORT} → {e}")
        return

    print("Connected. Sending poses one by one with a delay of {}s.\n".format(DELAY))

    for index, pose in enumerate(fixed_poses, start=1):
        cmd_str = format_move_command(pose)
        print(f"[{index}/{len(fixed_poses)}] Sending command: {cmd_str}")
        try:
            send_command_to_robot(sock, cmd_str)
        except Exception as e:
            print(f"  ERROR: Failed to send '{cmd_str}': {e}")
        # Wait for the robot’s “DONE” (or simply to complete) before moving on:
        time.sleep(DELAY)

    sock.close()
    print("\nAll {0} poses have been sent. Script complete.".format(len(fixed_poses)))


if __name__ == "__main__":
    main()
