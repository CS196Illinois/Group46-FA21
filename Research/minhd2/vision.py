import cv2
import time

cap = cv2.VideoCapture(1)
time.sleep(1.000)
cap.set(3,640)
cap.set(4,480)

while(True):
    success, frame = cap.read()
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break