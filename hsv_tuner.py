import cv2
import numpy as np

def nothing(x):
    pass

# Load ONE of your captured frames (Ensure the path matches where you saved them)
image = cv2.imread('dataset/raw_frames/frame_0050.jpg')

# 1. Apply Region of Interest (ROI) - Ignore the sky and the speedometer
height, width = image.shape[:2]
roi = image[int(height*0.4):int(height*0.8), :] 

cv2.namedWindow('Tuner')
cv2.createTrackbar('HMin', 'Tuner', 0, 179, nothing)
cv2.createTrackbar('SMin', 'Tuner', 0, 255, nothing)
cv2.createTrackbar('VMin', 'Tuner', 0, 255, nothing)
cv2.createTrackbar('HMax', 'Tuner', 179, 179, nothing)
cv2.createTrackbar('SMax', 'Tuner', 255, 255, nothing)
cv2.createTrackbar('VMax', 'Tuner', 255, 255, nothing)

# Set some initial guesses for dark gray/purple
cv2.setTrackbarPos('VMax', 'Tuner', 150)

while True:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    hMin = cv2.getTrackbarPos('HMin', 'Tuner')
    sMin = cv2.getTrackbarPos('SMin', 'Tuner')
    vMin = cv2.getTrackbarPos('VMin', 'Tuner')
    hMax = cv2.getTrackbarPos('HMax', 'Tuner')
    sMax = cv2.getTrackbarPos('SMax', 'Tuner')
    vMax = cv2.getTrackbarPos('VMax', 'Tuner')
    
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    cv2.imshow('Original ROI', roi)
    cv2.imshow('Mask (White = Road)', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Your Road HSV Bounds:\nLower: [{hMin}, {sMin}, {vMin}]\nUpper: [{hMax}, {sMax}, {vMax}]")
        break

cv2.destroyAllWindows()
