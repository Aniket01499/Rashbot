import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO

print("Loading Neural Network...")
# Pointing to the exact model you just trained
model = YOLO("runs/detect/train4/weights/best.pt")

sct = mss.mss()

# --- REPLACE THESE WITH YOUR EXACT xwininfo NUMBERS FROM EARLIER ---
monitor = {"top": 169, "left": 939, "width": 640, "height": 480} 
# -------------------------------------------------------------------

print("Vision System Online. Press 'q' in the video window to quit.")

# FPS Tracker variables
prev_time = 0

while True:
    # 1. Capture the screen
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # 2. Run the frame through the YOLOv8 Brain
    # verbose=False stops it from spamming your terminal
    results = model(frame, verbose=False) 
    
    # 3. Draw the bounding boxes on the frame
    annotated_frame = results[0].plot()
    
    # 4. Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 5. Show the live feed
    cv2.imshow("RashBot Active Vision", annotated_frame)
    
    # Emergency kill switch
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
