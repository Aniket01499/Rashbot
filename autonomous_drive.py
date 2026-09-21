import cv2
import subprocess
import numpy as np
import mss
import time
from ultralytics import YOLO

from planner import APFPlanner
from controller import SteeringController

print("Initializing RashBot ADAS (Darkness-Seeking Edition)...")

model = YOLO("runs/detect/train2/weights/best.pt")

planner = APFPlanner()
controller = SteeringController()

sct = mss.mss()
# Make sure your game window matches these coordinates exactly
monitor = {"top": 132, "left": 1185, "width": 640, "height": 480} 

prev_time = 0

try:
    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot() 
        boxes = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) > 0 else []
        
        # The critical unpacking line
        target_x, ego_x = planner.calculate_target_x(boxes, frame)
        
        if target_x is not None and ego_x is not None:
            controller.calculate_steering(target_x, ego_x)
            
            # Draw Telemetry
            cv2.line(annotated_frame, (target_x, 0), (target_x, planner.SCREEN_H), (0, 255, 0), 3)
            cv2.circle(annotated_frame, (ego_x, int(planner.BOT_Y)), 10, (255, 255, 0), -1)

        # HUD
        curr_time = time.time()
        fps = 1 / (max(curr_time - prev_time, 0.001)) # Safe divide
        prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RashBot Racing Dashcam", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Quit
        if key == ord('q'):
            break
            
        # Pause
        elif key == ord('p'):
            print("PAUSED: Dropping all inputs. Press 'p' to resume.")
            controller.emergency_stop() 
            controller.flush_keys() 
            
            while True:
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    print("RESUMING: Flushing OS Buffer and Refocusing Game...")
                    controller.flush_keys() 
                    
                    # --- NEW: Force Linux to make the game the active window! ---
                    # Change "Road Rash" to the exact window title of your game.
                    try:
                        subprocess.run(['xdotool', 'search', '--name', 'Default - Wine desktop', 'windowactivate'])
                    except Exception as e:
                        print("Could not refocus window. Is xdotool installed?")
                        
                    break

except Exception as e:
    print(f"System Crash: {e}")

finally:
    controller.emergency_stop()
    cv2.destroyAllWindows()