import cv2
import json
import numpy as np

# Define checkerboard size (internal corners)
CHECKERBOARD_SIZE = (4, 3)  

def capture_checkerboard():
    """Captures an image and detects the checkerboard corners."""
    cap = cv2.VideoCapture(1)  # Change to 1 if using an external camera
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Press 'c' to capture
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def detect_checkerboard(image):
    """Detects checkerboard corners in an image and returns their 2D pixel locations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    if found:
        # Refine corner locations
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Draw corner indices for verification
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            cv2.putText(image, str(i+1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display annotated checkerboard
        cv2.imshow("Annotated Checkerboard", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print detected points to ensure correct order
        print("\n✅ Detected Image Points Order:")
        for i, corner in enumerate(corners):
            print(f"Corner {i+1}: {corner.ravel()}")

        return corners
    else:
        print("Checkerboard not found.")
        return None

def load_gripper_coordinates():
    """Loads the manually recorded gripper 6D coordinates from a file."""
    with open("gripper_coordinates.json", "r") as file:
        return json.load(file)

def create_mapping(image_points, gripper_positions):
    """Creates a direct mapping between image coordinates and gripper 6D coordinates."""
    mapping = {}
    for i in range(len(image_points)):
        x_img, y_img = image_points[i].ravel()  # Convert to (x, y)
        key = f"{x_img},{y_img}"  # Convert tuple to a string key
        mapping[key] = gripper_positions[i]  # Store 6D gripper coordinates
    
    return mapping

def update_point_mapping():
    """
    Captures a new checkerboard image, detects its corners,
    and creates an updated mapping between the detected 2D image points and
    the previously recorded 6D gripper coordinates.
    """
    # Step 1: Capture an image of the checkerboard.
    print("\n[Mapping] Capturing checkerboard image. Press 'c' to capture, 'q' to quit.")
    image = capture_checkerboard()
    if image is None:
        print("Error: Could not capture checkerboard image.")
        return False

    # Step 2: Detect checkerboard corners.
    image_points = detect_checkerboard(image)
    if image_points is None:
        print("Error: Checkerboard corners not detected.")
        return False

    # Step 3: Load the recorded gripper 6D coordinates.
    gripper_positions = load_gripper_coordinates()

    # Step 4: Create mapping between the detected image points and the gripper coordinates.
    point_mapping = create_mapping(image_points, gripper_positions)

    # Step 5: Save the updated mapping to point_mapping.json.
    with open("point_mapping.json", "w") as file:
        json.dump(point_mapping, file, indent=4)
    
    print("\n✅ 2D Image -> 6D Gripper mapping updated successfully!")
    return True


if __name__ == "__main__":
    # Step 1: Capture an image of the checkerboard
    image = capture_checkerboard()

    # Step 2: Detect checkerboard corners
    image_points = detect_checkerboard(image)

    if image_points is not None:
        # Step 3: Load the recorded gripper coordinates
        gripper_positions = load_gripper_coordinates()

        # Step 4: Create a mapping from 2D image points to 6D gripper coordinates
        point_mapping = create_mapping(image_points, gripper_positions)

        print("\n✅ 2D Image -> 6D Gripper Mapping:")
        for k, v in point_mapping.items():
            print(f"Image: {k} -> Gripper: {v}")

        # Save the mapping for future use
        with open("point_mapping.json", "w") as file:
            json.dump(point_mapping, file, indent=4)
