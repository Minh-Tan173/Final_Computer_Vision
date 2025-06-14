import os
import pickle
from deepface import DeepFace

# Cấu hình
model_name = "VGG-Face"  # đổi thành "MobileFaceNet" nếu muốn chạy model thứ 2
face_root_folder = "D:/Computer_Vision/face"
output_pkl = f"db_{model_name}.pkl"

# Hàm lấy embedding
def extract_embeddings(root_folder, model_name):
    embeddings = {}

    for person_name in os.listdir(root_folder):
        person_folder = os.path.join(root_folder, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            try:
                rep = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=True)
                embeddings[img_path] = {
                    "name": person_name,
                    "embedding": rep[0]["embedding"]
                }
                print(f"[✓] {img_name} -> {person_name}")
            except Exception as e:
                print(f"[X] Bỏ qua {img_name} do lỗi: {e}")

    return embeddings

# Chạy trích xuất và lưu
print(f"[INFO] Đang trích xuất embeddings bằng {model_name}...")
db = extract_embeddings(face_root_folder, model_name)

with open(output_pkl, "wb") as f:
    pickle.dump(db, f)

print(f"[✅] Đã lưu vào {output_pkl} với {len(db)} ảnh.")
