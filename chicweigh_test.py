# =========================================
# CHICWEIGH - Multi-Angle Object Detection & Weight Estimation
# Captures 4 images ~1m distance for improved mass estimate
# Supports:
#   (1) Static image (image.jpg)
#   (2) Live USB Camera
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
focal_length = 780    # focal length in pixels
CAMERA_INDEX = 0      # USB camera index
ARDUINO_PORT = 'COM5'
NUM_IMAGES = 4        # Number of angles/images for estimation
# --------------------------------

# ----------- IMAGE CAPTURE -----------
def get_images(num_images=NUM_IMAGES):
    choice = input("📷 Use (1) Static image or (2) Live camera? [1/2]: ").strip()
    
    images = []

    if choice == "1":
        for i in range(num_images):
            filename = f"image_{i+1}.jpg"
            if not os.path.exists(filename):
                raise FileNotFoundError(f"❌ '{filename}' not found. Place images in folder.")
            img = cv2.imread(filename)
            if img is None:
                raise ValueError(f"❌ Could not read {filename}.")
            images.append(img)
        print(f"🖼️ Loaded {num_images} static images.")
        return images

    elif choice == "2":
        print(f"🎥 Opening camera (capture {num_images} images)...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("❌ Could not open camera.")

        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow(f"Capture Image {count+1} (SPACE to take, ESC to cancel)", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("❌ Capture cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == 32:  # SPACE
                filename = f"image_{count+1}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ Captured {filename}")
                images.append(frame.copy())
                count += 1

        cap.release()
        cv2.destroyAllWindows()
        return images
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

    # Optional: show detection
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
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

# ----------- WEIGHT ESTIMATION (MULTI-IMAGE) -----------
def estimate_weight_multi(masks):
    volumes = []
    for mask in masks:
        pixel_area = np.sum(mask > 0)
        object_area = pixel_area * (pixel_scale ** 2)
        volume = object_area * approx_thickness
        volumes.append(volume)
    avg_volume = sum(volumes) / len(volumes)
    mass_kg = avg_volume * rho
    return avg_volume, mass_kg
# ----------------------------------------

# ----------- MAIN PROGRAM -----------
def main():
    images = get_images(NUM_IMAGES)
    masks = []
    labels = []

    print("🧠 Detecting object in all images...")
    for idx, img in enumerate(images):
        mask, label, score = detect_object(img)
        masks.append(mask)
        labels.append(label)
        print(f"✅ Image {idx+1}: Detected {label} ({score*100:.2f}%)")

    # Use the most frequent label as the object name
    from collections import Counter
    label_count = Counter(labels)
    object_label = label_count.most_common(1)[0][0]

    print("⚖️ Estimating mass using multi-angle approach...")
    volume, mass = estimate_weight_multi(masks)

    print("=================================")
    print(f"📦 Estimated Volume: {volume:.6f} m³")
    print(f"⚖️ Estimated Mass: {mass:.2f} kg")
    print(f"🐾 Detected Object: {object_label}")
    print("=================================")

    # Send to Arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=2)
        time.sleep(2)
        arduino.write(f"{mass:.2f},{object_label}\n".encode())
        arduino.close()
        print(f"📤 Sent to Arduino: {mass:.2f}kg, {object_label}")
    except Exception as e:
        print(f"⚠️ Arduino send failed: {e}")

if __name__ == "__main__":
    main()
