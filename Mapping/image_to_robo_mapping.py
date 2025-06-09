import json
import numpy as np
import scipy.spatial

# This script is responsible for the 2d image to 6d robot coord mapping

def load_robot_coord_mapping():
    """Loads the 6D image-to-gripper mapping from JSON and converts keys back to tuples."""
    with open("point_mapping.json", "r") as file:
        data = json.load(file)
    
    # Convert string keys back to tuples for lookup
    point_mapping = {tuple(map(float, k.split(','))): v for k, v in data.items()}
    return point_mapping

def find_closest_gripper_point(image_point, point_mapping):
    """Finds the closest mapped gripper coordinate for a new image point."""
    image_points = np.array(list(point_mapping.keys()), dtype=np.float32)  # 2D image points
    gripper_points = np.array(list(point_mapping.values()), dtype=np.float32)  # 6D gripper positions

    # Find the closest mapped image point using KDTree
    tree = scipy.spatial.KDTree(image_points)
    _, idx = tree.query(image_point)  # Get nearest neighbor

    return gripper_points[idx]  # Return corresponding 6D gripper coordinate



if __name__ == "__main__":
    # Test loading and querying
    point_mapping = load_robot_coord_mapping()
    test_image_point = (16, 240)  # Example detected object in image
    mapped_coords = find_closest_gripper_point(test_image_point, point_mapping)
    print(f"\n✅ Test Mapping: Image Point {test_image_point} -> Gripper Coordinates {mapped_coords}")
