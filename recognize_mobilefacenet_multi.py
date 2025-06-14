import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# Load database đã trích từ InsightFace
with open("db_MobileFaceNet_Insight.pkl", "rb") as f:
    db = pickle.load(f)

# Hàm tính cosine similarity
def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

# Khởi tạo MobileFaceNet từ InsightFace
app = FaceAnalysis(name="buffalo_l")  # bạn cũng có thể thử "antelopev2"
app.prepare(ctx_id=0)  # ctx_id=0 để dùng CPU, nếu bạn có GPU thì đổi = 0

cap = cv2.VideoCapture(0)
print("[INFO] Nhận diện nhiều người – MobileFaceNet (insightface) – ESC để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding

        name = recognize(embedding, db)

        # Vẽ khung mặt và tên
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition - Insight MobileFaceNet", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
