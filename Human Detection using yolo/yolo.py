import cv2
from ultralytics import YOLO
import os

# --- SETTINGS ---
# Define the path to your test image
TEST_IMAGE_PATH = r"C:\\1VU\\Final Year Project\\Datasets\\Non Human\\image_9.jpg"

def run_human_detection(image_path):
    """
    Loads a pre-trained YOLOv8n model to detect humans in an image,
    and displays the result.
    """
    # 1. Load the official pre-trained YOLOv8n model
    # The model will be downloaded automatically on the first run.
    print("[INFO] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # 2. Check if the image exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at: {image_path}")
        return

    # 3. Run inference on the image
    print(f"[INFO] Running detection on {os.path.basename(image_path)}...")
    # The 'person' class in the COCO dataset is class index 0.
    # We can specify `classes=0` to only detect people.
    results = model(image_path, classes=0, conf=0.5) # conf is the confidence threshold

    # 4. Process and display the results
    # The `results` object contains all detection information.
    # We can use its built-in `plot()` method to get an image with boxes drawn on it.
    for r in results:
        print(f"[INFO] Found {len(r.boxes)} human(s) in the image.")
        
        # Get the image with bounding boxes and labels
        annotated_image = r.plot()

        # Display the output
        cv2.imshow("YOLOv8 Human Detection", annotated_image)
        cv2.waitKey(0)  # Wait for a key press to close the image window
        cv2.destroyAllWindows()

# --- Main runner ---
if __name__ == "__main__":
    run_human_detection(TEST_IMAGE_PATH)