import cv2
import numpy as np

# Define checkerboard dimensions (Change based on your checkerboard pattern)
CHECKERBOARD_SIZE = (3, 4)  # Number of internal corners (not squares)

def capture_and_detect_checkerboard():
    """Captures an image from the webcam and detects the checkerboard pattern."""
    cap = cv2.VideoCapture(1)  # Use 0 for the default webcam, 1 for external

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'c' to capture an image and detect the checkerboard.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        # Show the live camera feed
        cv2.imshow("Live Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Press 'c' to capture an image
            print("Capturing image and detecting checkerboard...")
            cap.release()
            cv2.destroyAllWindows()
            detect_checkerboard(frame)
            return
        elif key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_checkerboard(image):
    """Detects the checkerboard pattern in a given image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

    if found:
        print("Checkerboard detected!")
        # Refine the corner locations
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Draw the checkerboard corners
        cv2.drawChessboardCorners(image, CHECKERBOARD_SIZE, corners, found)
        cv2.imshow("Checkerboard Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Checkerboard not found. Try adjusting the camera angle or lighting.")

if __name__ == "__main__":
    capture_and_detect_checkerboard()
