import mss
import time
import os
import cv2
import numpy as np

# 1. Create the structured directory
os.makedirs('dataset/raw_frames', exist_ok=True)

sct = mss.mss()

# --- REPLACE THESE WITH YOUR xwininfo NUMBERS ---
monitor = {"top": 299, "left": 866, "width": 640, "height": 480} 
# ------------------------------------------------

print("Initializing High-Speed Sensor Array (3 FPS)...")
print("Recording started. Switch to the game window and play!")
count = 0
total_frames = 720 # 4 minutes at 3 FPS

try:
    while count < total_frames: 
        # Grab the raw pixels
        img = np.array(sct.grab(monitor))
        
        # Convert from BGRA to BGR
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Save to dataset folder
        filename = f"dataset/raw_frames/frame_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        
        count += 1
        if count % 30 == 0: # Print telemetry every 10 seconds
            print(f"Telemetry: Captured {count}/{total_frames} frames...")
            
        time.sleep(0.33) # 3 Frames Per Second
        
except KeyboardInterrupt:
    print("\nManual Override: Recording stopped.")

print(f"Data collection complete! {count} frames saved to dataset/raw_frames.")