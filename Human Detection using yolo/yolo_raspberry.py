import cv2
from ultralytics import YOLO
import time
import numpy as np

# --- SETTINGS ---
SOURCE = 0 # Use a webcam for testing. Change to your RTSP stream.
FRAME_SKIP = 5 # Process every 5th frame

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print(f"[ERROR] Could not open video source: {SOURCE}")
    exit()

# Variables for FPS calculation and frame skipping
frame_count = 0
prev_time = 0

# --- REAL-TIME DETECTION LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # --- Optimization 1: Frame Skipping ---
    if frame_count % FRAME_SKIP != 0:
        continue # Skip this frame

    # --- Run Detection ---
    # We only run the model on the frames that are not skipped
    results = model(frame, classes=0, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # --- Optimization 2: FPS Calculation ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Put the FPS text on the annotated frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("YOLOv8 Real-Time Human Detection (Optimized)", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()