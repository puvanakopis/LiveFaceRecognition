import cv2
import face_recognition
import os
import numpy as np

# ---------------- Load images ----------------
known_face_encodings = []
known_face_names = []
for filename in os.listdir('./images'):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join('./images', filename))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            known_face_encodings.append(face_recognition.face_encodings(rgb_img)[0])
            known_face_names.append(os.path.splitext(filename)[0])
        except IndexError:
            print(f"No face found in {filename}, skipping.")



# ---------------- Initialize webcam ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale back up
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()