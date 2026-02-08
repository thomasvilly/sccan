import cv2

# Use DirectShow (Windows backend)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 1. FORCE 1080p (This is CRITICAL for wide angle)
# If this fails, the camera defaults to 640x480 which = ZOOMED IN CROP
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 2. RESET ZOOM (Force it to 100/Minimum)
# On Windows/Logitech, 100 is usually "1x" (No Zoom)
cap.set(cv2.CAP_PROP_ZOOM, 100)

# Open the Settings Window so you can fine-tune focus/exposure
cap.set(cv2.CAP_PROP_SETTINGS, 1)

# Create a window that fits on your screen
cv2.namedWindow("Tuning Mode", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuning Mode", 960, 540)

print("--- DIAGNOSTICS ---")
print(f"Actual Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
print("If this says 640x480, your USB port might be too slow for 1080p.")
print("-------------------")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
        
    cv2.imshow("Tuning Mode", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()