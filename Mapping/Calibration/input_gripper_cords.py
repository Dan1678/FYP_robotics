import json

# Define the checkerboard size (internal corners)
CHECKERBOARD_SIZE = (4, 3)  

def input_gripper_coords():
    """Manually input the recorded gripper 6D coordinates and save them to a file."""
    gripper_coords = []

    print("\nEnter the gripper robot's 6D coordinates for each checkerboard corner.")
    print("Use format: X Y Z Pitch Roll Yaw (e.g., 200 50 150 0 90 0)\n")

    for i in range(CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1]):
        coords = input(f"Enter gripper 6D coordinates for corner {i+1}: ")
        x, y, z, pitch, roll, yaw = map(float, coords.split())  # Convert input to floats
        gripper_coords.append([x, y, z, pitch, roll, yaw])

    # Save to a JSON file
    with open("gripper_coordinates.json", "w") as file:
        json.dump(gripper_coords, file, indent=4)

    print("\n✅ Gripper 6D coordinates saved successfully!")
    return gripper_coords

if __name__ == "__main__":
    input_gripper_coords()
