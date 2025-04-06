import pickle
from insightface.app import FaceAnalysis
import cv2
from pathlib import Path

folder = Path(r'D:\Attendence Face detection\dataset')
img_extention = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']

img_file = [file for file in folder.iterdir() if file.suffix.lower() in img_extention]
img_number = len(img_file)
file_name = [file.stem for file in img_file]

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
registered_faces = []

saved_faces = 'dataset/saved_faces.pkl'

for f_count in range(1, img_number+1):
    face_1 = f'dataset/{file_name[f_count -1]}.jpg'
    img = cv2.imread(face_1)

    faces = app.get(img)
    for face in faces:
           x1, y1, x2, y2 = face.bbox.astype(int)
           cropped = img[y1:y2, x1:x2]
           embedding = face.embedding.reshape(1, -1)
           name = file_name[f_count -1]
           registered_faces.append((name, embedding))

with open(saved_faces, 'wb') as f:
            pickle.dump(registered_faces, f)
