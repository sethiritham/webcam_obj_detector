import cv2
import torch
import numpy as np
import torchvision
import time
from PIL import Image
from sort_tracker.sort import Sort

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1)*max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou_score = interArea / float(boxAArea + boxBArea - interArea)

    return iou_score

weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)


device = ("cuda" if torch.cuda.is_available() else "cpu")
url = "http://10.66.1.64:4747/video"
model.to(device)
model.eval()
preprocess = weights.transforms()
coco_names = weights.meta["categories"]

mot_tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

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
track_bbs_ids = np.empty((0,5))


track_data = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame.")
        break

    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    frame_count += 1

    # --- This set will hold all track IDs seen in this frame ---
    current_track_ids = set()

    if frame_count % process_every_n_frames == 0:
        # --- DETECTION FRAME ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        img_tensor = preprocess(pil_image).to(device)
        with torch.no_grad():
            pred = model([img_tensor])[0]

        detections_list = [] # For the tracker [x1, y1, x2, y2, score]
        full_detections = [] # For us [box, score, class_name]

        for i in range(len(pred["boxes"])): # Use pred["boxes"] not last_boxes
            score = pred["scores"][i].item() # Use pred["scores"]
            if score < 0.5:
                continue

            box = list(map(int, pred["boxes"][i].cpu().numpy().tolist()))
            label_index = pred["labels"][i].item() # Use pred["labels"]
            class_name = coco_names[label_index] if 0 <= label_index < len(coco_names) else "Unknown"

            detections_list.append([box[0], box[1], box[2], box[3], score])
            full_detections.append({
                "box" : box,
                "score" : score,
                "class_name" : class_name
            })
        
        if len(detections_list) > 0:
            detections_np = np.array(detections_list)
        else:
            detections_np = np.empty((0,5))
            
        # Update tracker
        track_bbs_ids = mot_tracker.update(detections_np)

        # --- Associate new track IDs with their class and score ---
        new_track_data = {}
        for track in track_bbs_ids:
            track_id = int(track[4])
            current_track_ids.add(track_id)
            
            # If this is a new track, find its data
            if track_id not in track_data:
                track_box = list(map(int, track[:4]))
                best_iou = 0
                best_match = None
                for det in full_detections:
                    current_iou = iou(track_box, det["box"])
                    # Use the tracker's iou_threshold
                    if current_iou > mot_tracker.iou_threshold: 
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_match = det
                
                if best_match:
                    new_track_data[track_id] = {
                        "class_name": best_match["class_name"],
                        "score": best_match["score"]
                    }
        
        # --- FIX: Update data ONCE, after the loop ---
        track_data.update(new_track_data)

    else:
        # --- SKIPPED FRAME ---
        predicted_boxes = mot_tracker.update(np.empty((0, 5)))
        
        # Flicker fix
        if len(predicted_boxes) > 0:
            track_bbs_ids = predicted_boxes
        
        # Add these predicted IDs to the current set
        for track in track_bbs_ids:
            current_track_ids.add(int(track[4]))

    # --- Clean up old tracks from our data dictionary ---
    # This is the fix for the "(lost)" text
    all_known_ids = set(track_data.keys())
    for old_id in all_known_ids:
        if old_id not in current_track_ids:
            # This ID is no longer being tracked by SORT, so we can delete its data
            del track_data[old_id]

    # --- CORRECTED DRAWING LOOP ---
    for i in range (len(track_bbs_ids)):
        box = track_bbs_ids[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        track_id = int(box[4]) # <-- This is the correct integer ID

        # Get the persistent data for this track
        info = track_data.get(track_id)

        if info:
            # Data found! Use it.
            class_name = info["class_name"]
            score = info["score"]
            text = f"{track_id}: {class_name} ({score:.2f})"
        else:
            # This should only happen for 1-2 frames if a track is brand new
            text = f"ID: {track_id}" 

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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