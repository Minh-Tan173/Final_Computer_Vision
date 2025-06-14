import cv2
import os

# === CẤU HÌNH ===
person_name = "HongAnh"  # "Hoang"
save_dir = f"D:/Computer_Vision/face/{person_name}"
os.makedirs(save_dir, exist_ok=True)

# Mở webcam
cap = cv2.VideoCapture(0)
img_count = 2
max_images = 100 # Số ảnh muốn chụp

print("[INFO] Nhấn SPACE để chụp, ESC để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hiển thị frame
    cv2.imshow("Webcam - Chụp ảnh", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC để thoát
        break
    elif key == 32:  # SPACE để chụp
        img_path = os.path.join(save_dir, f"{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[✓] Đã lưu: {img_path}")
        img_count += 1

        if img_count >= max_images:
            print("[INFO] Đã chụp đủ ảnh")
            break

cap.release()
cv2.destroyAllWindows()
