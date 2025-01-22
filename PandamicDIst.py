import numpy as np
import cv2

face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

cap = cv2.VideoCapture("./videos/video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    person_coords = []

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 250), 2)
        person_coords.append((x + w//2, y + h//2))
    
    for i in range(len(person_coords)):
        for j in range(i + 1, len(person_coords)):
            dist = calculate_distance(person_coords[i], person_coords[j])
            if dist < 100:
                cv2.imwrite('./photos/saved_frame.jpg', frame)
                cv2.putText(frame, "Alert! Too Close!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255),4)
    
    cv2.imshow('Social Distancing Monitoring', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
