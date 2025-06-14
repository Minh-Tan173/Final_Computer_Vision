import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import cv2

# Cấu hình
face_root_folder = "D:/Computer_Vision/face"
output_pkl = "db_MobileFaceNet_Insight.pkl"

# Khởi tạo InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # CPU

embeddings = {}

for person_name in os.listdir(face_root_folder):
    folder_path = os.path.join(face_root_folder, person_name)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (640, 480))  # resize để tăng khả năng detect

            faces = app.get(img)
            if faces:
                embedding = faces[0].embedding.tolist()
                embeddings[img_path] = {
                    "name": person_name,
                    "embedding": embedding
                }
                print(f"[✓] {img_name} → {person_name}")
            else:
                print(f"[X] Không thấy mặt trong {img_name}")
        except Exception as e:
            print(f"[X] Lỗi ở {img_name}: {e}")

with open(output_pkl, "wb") as f:
    pickle.dump(embeddings, f)

print(f"[✅] Đã lưu thành công → {output_pkl}")
