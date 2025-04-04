import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite("Captured_faces\captured_face.jpg", face_img)

    
    cv2.imshow("Face Detector", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
