import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# CONFIGURATION
# yolov8n
model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)

# FORCE standard resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

DROP_ZONE = (50, 50, 200, 200) 
ALLOWED_CLASSES = [39, 41, 67, 73, 76] # bottle, cup, phone, laptop, vase
CONF_THRESHOLD = 0.25 
SMOOTHING = 0.15 

# STATE
state = "IDLE"
pick_queue = deque()
active_item = None 
completed_count = 0

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    
    # 1. LIGHTWEIGHT ENHANCEMENT
    enhanced = apply_clahe(frame)
    display_frame = frame.copy()

    # 2. SINGLE PASS DETECTION (Faster & detects close objects better)
    # We use 640 here to keep the FPS high and lag low
    results = model(enhanced, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]

    all_dets = []
    for i, b in enumerate(results.boxes):
        cls = int(b.cls[0])
        if cls in ALLOWED_CLASSES:
            box = b.xyxy[0].cpu().numpy()
            all_dets.append({
                'id': i + 1, 
                'box': box, 
                'label': model.names[cls],
                'center': ((box[0]+box[2])//2, (box[1]+box[3])//2)
            })

    # 3. DRAW UI
    cv2.rectangle(display_frame, (DROP_ZONE[0], DROP_ZONE[1]), (DROP_ZONE[2], DROP_ZONE[3]), (255, 255, 0), 2)
    cv2.putText(display_frame, "DROP ZONE", (DROP_ZONE[0], DROP_ZONE[1]-10), 0, 0.6, (255, 255, 0), 2)
    cv2.putText(display_frame, f"QUEUE: {len(pick_queue)} | DONE: {completed_count}", (20, h-20), 0, 0.6, (255, 255, 255), 2)

    # 4. VISUALIZE IDLE STATE
    if state == "IDLE":
        for det in all_dets:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"ID:{det['id']} {det['label']}", (x1, y1-10), 0, 0.5, (0, 255, 0), 2)

    # 5. PICK & PLACE LOGIC
    if (state == "PICKING" or state == "PLACING") and active_item:
        # Movement
        active_item['cx'] += (active_item['tx'] - active_item['cx']) * SMOOTHING
        active_item['cy'] += (active_item['ty'] - active_item['cy']) * SMOOTHING
        
        curr_p = (int(active_item['cx']), int(active_item['cy']))
        
        # Draw "Gripper"
        cv2.circle(display_frame, curr_p, 15, (0, 165, 255), -1)
        cv2.putText(display_frame, f"CARRYING ID:{active_item['id']}", (curr_p[0]+20, curr_p[1]), 0, 0.6, (0, 165, 255), 2)

        # Distance to target
        dist = np.hypot(active_item['cx'] - active_item['tx'], active_item['cy'] - active_item['ty'])
        if dist < 5:
            if state == "PICKING":
                state = "PLACING"
                active_item['tx'], active_item['ty'] = (DROP_ZONE[0]+DROP_ZONE[2])//2, (DROP_ZONE[1]+DROP_ZONE[3])//2
            else:
                completed_count += 1
                active_item = None
                if pick_queue:
                    active_item = pick_queue.popleft()
                    state = "PICKING"
                else:
                    state = "IDLE"

    # 6. CONTROLS
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p') and state == "IDLE":
        for det in all_dets:
            cx, cy = det['center']
            pick_queue.append({'id': det['id'], 'cx': cx, 'cy': cy, 'tx': cx, 'ty': cy, 'label': det['label']})
        if pick_queue:
            active_item = pick_queue.popleft()
            state = "PICKING"
            
    if key == ord('r'):
        state, pick_queue, active_item, completed_count = "IDLE", deque(), None, 0

    cv2.imshow("MSc Project - Stable Pick & Place", display_frame)
    if key == 27: break

cap.release()
cv2.destroyAllWindows()