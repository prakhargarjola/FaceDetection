from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
import os
import pickle

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

os.makedirs("registered_faces", exist_ok=True)
embeddings_file = "registered_faces/face_embeddings.pkl"
registered_faces = []
face_count = 0

if os.path.exists(embeddings_file) and os.path.getsize(embeddings_file) > 0:
    try:
        with open(embeddings_file, "rb") as f:
            registered_faces = pickle.load(f)
        face_count = len(registered_faces)
    except:
        registered_faces = []
        face_count = 0

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cropped = frame[y1:y2, x1:x2]
        embedding = face.embedding.reshape(1, -1)

        is_duplicate = False
        for reg_face in registered_faces:
            sim = cosine_similarity(embedding, reg_face)[0][0]
            if sim > 0.6:
                is_duplicate = True
                break

        if not is_duplicate:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "New face - Press 's'", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                image_path = f"registered_faces/face_{face_count}.jpg"
                cv2.imwrite(image_path, cropped)
                registered_faces.append(embedding)
                with open(embeddings_file, "wb") as f:
                    pickle.dump(registered_faces, f)
                face_count += 1
        else:
            cv2.putText(frame, "Already registered", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Face Registration", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
