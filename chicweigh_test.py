# =========================================
# CHICWEIGH - Single Camera Object Detection & Weight Estimation
# Supports:
#   (1) Static image (image.jpg)
#   (2) Live USB Camera (COM4)
# =========================================

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import serial
import time
import os

# ----------- SETTINGS -----------
rho = 1050            # object density (kg/m³)
pixel_scale = 0.0035  # 1 pixel ≈ 3.5 mm
approx_thickness = 0.04  # assumed object thickness (m)
focal_length = 780    # focal length in pixels (for camera scaling if needed)

CAMERA_INDEX = 0      # Usually 0 for USB cameras
ARDUINO_PORT = 'COM5'
# --------------------------------


# ----------- IMAGE SOURCE SELECTION -----------
def get_image():
    choice = input("📷 Use (1) Static image or (2) Live camera? [1/2]: ").strip()

    if choice == "1":
        if not os.path.exists("image.jpg"):
            raise FileNotFoundError("❌ 'image.jpg' not found. Place an image in the same folder.")
        print("🖼️ Using static image: image.jpg")
        img = cv2.imread("image.jpg")
        if img is None:
            raise ValueError("❌ Could not read image.jpg. Check file integrity.")
        return img

    elif choice == "2":
        print("🎥 Opening camera (Felta USB COM4)...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("❌ Could not open camera. Try replugging your Felta device.")

        print("🎬 Press SPACE to capture image, or ESC to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No frame captured.")
                continue

            cv2.imshow("Felta Camera (COM4)", frame)
            key = cv2.waitKey(1)

            if key == 27:  # ESC
                print("❌ Capture cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == 32:  # SPACE
                cv2.imwrite("image.jpg", frame)
                print("✅ Image captured and saved as image.jpg.")
                img = frame.copy()
                break

        cap.release()
        cv2.destroyAllWindows()
        return img
    else:
        raise ValueError("⚠️ Invalid choice. Enter 1 or 2.")
# ------------------------------------


# ----------- OBJECT DETECTION -----------
def detect_object(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    if len(outputs[0]['scores']) == 0:
        raise Exception("❌ No objects detected.")

    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    masks = outputs[0]['masks'].cpu().numpy()

    best_idx = np.argmax(scores)
    best_mask = (masks[best_idx, 0] > 0.5).astype(np.uint8) * 255
    best_label = labels[best_idx]
    best_score = scores[best_idx]

    coco_classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    label_name = coco_classes[best_label] if best_label < len(coco_classes) else "unknown"

    # Draw bounding box
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(result, f"{label_name} ({best_score*100:.1f}%)", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Detection Result", result)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return best_mask, label_name, best_score
# ----------------------------------------


# ----------- WEIGHT ESTIMATION -----------
def estimate_weight(mask):
    pixel_area = np.sum(mask > 0)
    object_area = pixel_area * (pixel_scale ** 2)
    object_volume = object_area * approx_thickness
    mass_kg = object_volume * rho
    return object_volume, mass_kg
# ----------------------------------------


# ----------- MAIN PROGRAM -----------
def main():
    img = get_image()
    print("🧠 Detecting object...")
    mask, label, confidence = detect_object(img)
    print(f"✅ Detected: {label} ({confidence*100:.2f}%)")

    print("⚖️ Estimating weight...")
    volume, mass = estimate_weight(mask)
    print("=================================")
    print(f"📦 Estimated Volume: {volume:.6f} m³")
    print(f"⚖️ Estimated Weight: {mass:.2f} kg")
    print("=================================")

    try:
        arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=2)
        time.sleep(2)
        arduino.write(f"{mass:.2f},{label}\n".encode())
        arduino.close()
        print(f"📤 Sent to Arduino: {mass:.2f}kg, {label}")
    except Exception as e:
        print(f"⚠️ Arduino send failed: {e}")

if __name__ == "__main__":
    main()
