import cv2

def show_camera():
    """Opens the computer's webcam to help determine a good camera position."""
    cap = cv2.VideoCapture(1) 

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to close the camera window.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Camera View", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera()
