import cv2
import pickle
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace

# Load DB đã lưu
with open("db_VGG-Face.pkl", "rb") as f:
    db = pickle.load(f)

# Tính cosine similarity
def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# So sánh embedding với DB → trả về tên và điểm similarity cao nhất
def recognize(embedding, db, threshold=0.6):
    best_name = "Unknown"
    best_score = -1
    for info in db.values():
        score = cosine_sim(embedding, info["embedding"])
        if score > best_score:
            best_score = score
            best_name = info["name"]
    if best_score > threshold:
        return f"{best_name} ({best_score:.2f})"
    else:
        return "Unknown"

# Webcam
cap = cv2.VideoCapture(0)

print("[INFO] Nhận diện nhiều người – VGG-Face – ESC để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Dò tất cả khuôn mặt
        faces = RetinaFace.detect_faces(frame)

        if isinstance(faces, dict):
            for face_key in faces:
                face = faces[face_key]
                x, y, w, h = face["facial_area"]
                face_crop = frame[y:h, x:w]

                # Trích đặc trưng
                rep = DeepFace.represent(face_crop, model_name="VGG-Face", enforce_detection=False)
                embedding = rep[0]["embedding"]

                # Gán nhãn
                label = recognize(embedding, db)

                # Vẽ khung + tên
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        cv2.putText(frame, "Lỗi nhận diện", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition - Multi (VGG-Face)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
