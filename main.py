import cv2
import torch
import torchvision
import time
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# Load model
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()
preprocess = weights.transforms()
coco_names = weights.meta["categories"]

TARGET_WIDTH, TARGET_HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam opened successfully!")

prev_frame_time = 0
frame_count = 0
process_every_n_frames = 3
last_boxes, last_labels, last_scores = [], [], []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    frame_count += 1

    if frame_count % process_every_n_frames == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        img_tensor = preprocess(pil_image).to(device)
        with torch.no_grad():
            pred = model([img_tensor])[0]
        last_boxes, last_labels, last_scores = pred["boxes"], pred["labels"], pred["scores"]

    if last_boxes is not None and len(last_boxes) > 0:
        for i in range(len(last_boxes)):
            score = last_scores[i].item()
            if score < 0.5:
                continue

            x1, y1, x2, y2 = map(int, last_boxes[i].cpu().numpy().tolist())
            label_index = last_labels[i].item()
            class_name = coco_names[label_index] if 0 <= label_index < len(coco_names) else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
