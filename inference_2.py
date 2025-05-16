import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(
    "/home/prakhar/Desktop/College/3rdYear/SEM-VI/DSML/Sign_Language/runs/detect/train/weights/best.pt")

# Open the camera
camera = cv2.VideoCapture(1)  # Change index to 0 if using default webcam

if not camera.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Set camera resolution (increase capture area)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increase height

# Create a resizable window
cv2.namedWindow("YOLO Live Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Live Inference", 1280, 720)  # Set window size

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model(frame)  # Run YOLO model on frame
    annotated_frame = results[0].plot()  # Get visualized predictions

    cv2.imshow("YOLO Live Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
