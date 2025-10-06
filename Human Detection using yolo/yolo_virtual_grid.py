import cv2
from ultralytics import YOLO
import time
import numpy as np

# --- SETTINGS ---
SOURCE = 0 # Use a webcam.
DURATION = 5 # Run the detection for 5 seconds.
FRAME_SKIP = 5 # Process every 5th frame

# --- VIRTUAL GRID AND APPLIANCE MAPPING ---
GRID_ROWS = 3
GRID_COLS = 3
appliance_map = {
    (0, 0): ["Light 1 (Back Left)"], (0, 1): ["Light 2 (Back Center)"], (0, 2): ["Light 3 (Back Right)"],
    (1, 0): ["Fan 1 (Mid Left)"],    (1, 1): ["Projector"],             (1, 2): ["Fan 2 (Mid Right)"],
    (2, 0): ["Light 4 (Front Left)"], (2, 1): ["Light 5 (Front Center)"],(2, 2): ["Light 6 (Front Right)"],
}

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print(f"[ERROR] Could not open video source: {SOURCE}")
    exit()

# --- INITIALIZATION ---
frame_count = 0
prev_time = time.time()
start_time = time.time() # Start the 5-second timer

human_detected_at_all = False
last_occupied_grids = set()

print(f"[INFO] Starting {DURATION}-second scan for human presence...")

# --- DETECTION LOOP (RUNS FOR 5 SECONDS) ---
while (time.time() - start_time) < DURATION:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions for grid calculation
    h, w, _ = frame.shape
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS
    
    annotated_frame = frame.copy()
    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        results = model(frame, classes=0, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        
        current_frame_grids = set()
        
        for r in results:
            if len(r.boxes) > 0:
                human_detected_at_all = True # Set flag if a human is ever seen
            
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                grid_col, grid_row = int(center_x // cell_w), int(center_y // cell_h)
                
                if 0 <= grid_row < GRID_ROWS and 0 <= grid_col < GRID_COLS:
                    current_frame_grids.add((grid_row, grid_row))
        
        # Always update with the most recent detection state
        last_occupied_grids = current_frame_grids

    # --- VISUAL FEEDBACK (Drawing grid and FPS) ---
    for i in range(1, GRID_COLS): cv2.line(annotated_frame, (i * cell_w, 0), (i * cell_w, h), (0, 255, 255), 1)
    for i in range(1, GRID_ROWS): cv2.line(annotated_frame, (0, i * cell_h), (w, i * cell_h), (0, 255, 255), 1)
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Zonal Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- FINAL COMMAND LOGIC (RUNS ONCE AFTER 5 SECONDS) ---
print("\n--- 5-Second Scan Complete ---")

if human_detected_at_all:
    print("[RESULT] Human presence was detected. Issuing final commands...")
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            zone = (r, c)
            appliances = appliance_map.get(zone, [])
            if not appliances: continue

            if zone in last_occupied_grids:
                print(f"[COMMAND] Zone {zone} is OCCUPIED. Turning ON: {', '.join(appliances)}")
                # TODO: Add RPi.GPIO logic to turn ON relays
            else:
                print(f"[COMMAND] Zone {zone} is EMPTY. Turning OFF: {', '.join(appliances)}")
                # TODO: Add RPi.GPIO logic to turn OFF relays
else:
    print("[RESULT] No human presence detected. All appliances will remain OFF.")
    # Optional: Add GPIO logic here to ensure all relays are OFF

# --- CLEANUP ---
print("\n[INFO] Closing program.")
cap.release()
cv2.destroyAllWindows()