import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.realpath(os.path.join(current_dir, '../')))

import cv2
from SC2.utils.convert import cv_to_pil_image, pil_to_cv_image
from SC2.mtcnn.mtcnn import MTCNN
from SC2.arcface.arcface import ArcFace

SKIP_FRAME = 2


data_dir = os.path.realpath(os.path.join(current_dir, 'sample'))
files0 = os.listdir(data_dir)
n_files = len(files0)
files = []
directories = []
for file in files0:
    files.append(
        os.path.realpath(
            os.path.join(data_dir, file))
    )
    directories.append(
        os.path.realpath(
            os.path.join(data_dir, os.path.splitext(file)[0])
        )
    )

mtcnn = MTCNN()
arcface = ArcFace()

for i in range(n_files):
    if not os.path.exists(directories[i]):
        os.makedirs(directories[i])
    cap = cv2.VideoCapture(files[i])
    if cap.isOpened() is False:
        continue
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            image = cv_to_pil_image(frame)
            image = image.convert(mode='RGB')
            image = image.rotate(-90, expand=1)
            width, height = image.size
            image_size = width * height
            bounding_boxes, landmarks, faces = mtcnn.align(image, min_face_size=20, thresholds=[0.7, 0.7, 0.8])
            if len(faces) == 0:
                continue
            face_index = 0
            for j in range(len(faces)):
                x1, y1, x2, y2, score = bounding_boxes[j, 0], bounding_boxes[j, 1], bounding_boxes[j, 2], bounding_boxes[j, 3], bounding_boxes[j, 4]
                # ---------------------------------
                
                # ---------------------------------
                image = pil_to_cv_image(faces[j])
                cv2.imwrite(
                    os.path.realpath(
                        os.path.join(
                            directories[i],
                            'face{}frame{}.jpg'.format(face_index, frame_index)
                        )
                    ),
                    image
                )
                face_index += 1
            if face_index > 0:
                frame_index += 1
        else:
            break
    
