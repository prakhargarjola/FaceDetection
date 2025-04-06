import pickle
from insightface.app import FaceAnalysis
import cv2
from pathlib import Path

folder = Path(r'D:\Attendence Face detection\dataset')
img_extention = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

img_file = [file for file in folder.iterdir() if file.suffix.lower() in img_extention]
img_number = len(img_file)



app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
registered_faces = []

saved_faces = 'dataset/saved_faces.pkl'

for f_count in range(img_number):
    face_1 = f'dataset/{f_count}.jpg'
    img = cv2.imread(face_1)

    faces = app.get(img)
    for face in faces:
        embedding = face.embedding.reshape(1, -1)
        registered_faces.append(embedding)


with open(saved_faces, 'wb') as f:
            pickle.dump(registered_faces, f)
