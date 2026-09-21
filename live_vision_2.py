import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO

print("Initializing Live Vision Debugger (75 Epoch Model)...")

# Loading the NEW model from the train2 directory
model = YOLO("runs/detect/train2/weights/best.pt")

sct = mss.mss()
# Your exact game window coordinates
monitor = {"top": 132, "left": 1185, "width": 640, "height": 480} 

prev_time = 0

try:
    while True:
        # Capture screen
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Run YOLO inference
        results = model(frame, verbose=False)
        
        # Draw YOLO's detections onto the frame
        annotated_frame = results[0].plot() 

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the video feed
        cv2.imshow("RashBot Vision - 75 Epochs", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Debugger crashed: {e}")

finally:
    cv2.destroyAllWindows()
