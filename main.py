import cv2
import pickle
import time
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Load DB
with open("db_VGG-Face.pkl", "rb") as f:
    db_vgg = pickle.load(f)
with open("db_MobileFaceNet_Insight.pkl", "rb") as f:
    db_mobile = pickle.load(f)

# Load InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# Hàm cosine
def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Hàm nhận diện cho từng DB
def recognize(embedding, db, threshold=0.6):
    best_name = "Unknown"
    best_score = -1
    for info in db.values():
        score = cosine_sim(embedding, info["embedding"])
        if score > best_score:
            best_score = score
            best_name = info["name"]
    if best_score > threshold:
        return best_name, best_score
    else:
        return "Unknown", best_score

# Webcam
cap = cv2.VideoCapture(0)

print("[INFO] So sánh VGG-Face vs MobileFaceNet – ESC để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_crop = frame[y1:y2, x1:x2]

        # VGG-Face
        t1 = time.time()
        try:
            rep_vgg = DeepFace.represent(face_crop, model_name="VGG-Face", enforce_detection=False)
            emb_vgg = rep_vgg[0]["embedding"]
            name_vgg, score_vgg = recognize(emb_vgg, db_vgg)
        except:
            name_vgg, score_vgg = "Error", -1
        t2 = time.time()

        # MobileFaceNet
        t3 = time.time()
        try:
            emb_mobile = face.embedding
            name_mobile, score_mobile = recognize(emb_mobile, db_mobile)
        except:
            name_mobile, score_mobile = "Error", -1
        t4 = time.time()

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"VGG: {name_vgg} ({score_vgg:.2f})", (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(frame, f"MOB: {name_mobile} ({score_mobile:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # In console
        print("="*40)
        print(f"[VGG-Face]       → {name_vgg} | Cosine: {score_vgg:.2f} | Time: {(t2 - t1):.3f}s")
        print(f"[MobileFaceNet]  → {name_mobile} | Cosine: {score_mobile:.2f} | Time: {(t4 - t3):.3f}s")

    cv2.imshow("Face Comparison – VGG vs MobileFaceNet", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
