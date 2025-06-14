
# Robust Face Recognition: A Comparative Analysis of VGG-Face and MobileFaceNet

## 📌 Abstract
This project presents a comparative study of two face recognition models — **VGG-Face** (via DeepFace) and **MobileFaceNet** (via InsightFace) — applied on a custom dataset collected via webcam. The system extracts and compares facial embeddings using cosine similarity, evaluates real-time recognition capability, and provides interactive visualizations via Streamlit. Results highlight the trade-offs between accuracy and computational efficiency across the two models.

---

## 1. Introduction
Face recognition is a critical task in modern computer vision, with applications ranging from security systems to user authentication. This project explores and contrasts the performance of two popular architectures:

- **VGG-Face**: a high-accuracy but computationally intensive model.
- **MobileFaceNet**: a lightweight model suitable for real-time deployment.

We evaluate both using cosine similarity on embedded vectors and assess recognition accuracy in live webcam scenarios.

---

## 2. Dataset
- 📁 **Source**: Manually captured via webcam (`take_picture.py`)
- 👤 **Format**: Stored under `D:/Computer_Vision/face/<person_name>/image.jpg`
- 📷 **Data Collection Tool**: A simple OpenCV webcam interface
- 📊 **Content**: Face images from multiple individuals for training/testing

---

## 3. Methodology

### 3.1. Embedding Extraction
- `generate_db.py`: Uses **DeepFace** to extract embeddings via VGG-Face
- `generate_db_insight_mobile.py`: Uses **InsightFace** to extract MobileFaceNet embeddings

Each script serializes the result into `.pkl` databases containing:
```python
{
  "image_path": {
    "name": "PersonName",
    "embedding": [128/512-dim vector]
  }
}
```

### 3.2. Real-Time Recognition
- `main.py`: Runs real-time face recognition via webcam using both models.
- Each detected face is matched against the embedding database using cosine similarity:
  
  ```python
  sim = cosine(a, b) = dot(a, b) / (||a|| * ||b||)
  ```

### 3.3. Model-specific Recognition Scripts
- `recognize_vggface.py`: Standalone VGG-Face-based multi-face recognition
- `recognize_mobilefacenet_multi.py`: Real-time MobileFaceNet recognition using InsightFace

---

## 4. Visualization and Analysis

### 4.1. Streamlit Dashboard
- Script: `ShowDataAndResult.py`
- Features:
  - Dataset summary
  - Embedding visualization
  - Pairwise similarity matrix
  - Comparative performance (VGG vs MobileFaceNet)
  - Analytical comments for mismatches or misidentifications

---

## 5. Results

### Quantitative Comparison (as shown in Streamlit app)
- ✅ VGG-Face performed better on X/Y same-person comparisons
- ⚡ MobileFaceNet outperformed in terms of speed and handled multiple faces in real time

| Model          | Accuracy | Speed    | Real-Time Suitability |
|----------------|----------|----------|------------------------|
| VGG-Face       | High     | Slow     | ❌                     |
| MobileFaceNet  | Moderate | Fast     | ✅                     |

### Observations
- VGG-Face achieves better identity preservation, but with higher latency
- MobileFaceNet is more efficient and stable for lightweight applications

---

## 6. Installation & Usage

### 6.1. Requirements
```bash
pip install deepface insightface streamlit opencv-python
```

### 6.2. Data Collection
```bash
python take_picture.py  # Save images to dataset
```

### 6.3. Generate Databases
```bash
python generate_db.py              # For VGG-Face
python generate_db_insight_mobile.py  # For MobileFaceNet
```

### 6.4. Real-Time Recognition
```bash
python main.py
```

### 6.5. Visualization
```bash
streamlit run ShowDataAndResult.py
```

---

## 7. Folder Structure

```
.
├── generate_db.py
├── generate_db_insight_mobile.py
├── main.py
├── recognize_mobilefacenet_multi.py
├── recognize_vggface.py
├── take_picture.py
├── ShowDataAndResult.py
├── db_VGG-Face.pkl
├── db_MobileFaceNet_Insight.pkl
└── D:/Computer_Vision/face/
    └── <PersonName>/
        └── *.jpg
```

---

## 8. Limitations and Future Work
- ❌ No face alignment or illumination normalization
- ❌ Database size limited to webcam-collected images
- 🔜 Future extensions:
  - Apply face alignment using dlib or mediapipe
  - Integrate anti-spoofing
  - Expand dataset size and class diversity
  - Incorporate fine-tuned MobileNet or ArcFace models

---

## 9. References

```bibtex
@article{Parkhi2015VGGFace,
  title={Deep Face Recognition},
  author={Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman},
  journal={BMVC},
  year={2015}
}

@article{Deng2020InsightFace,
  title={InsightFace: Deep Face Analysis Toolkit},
  author={Jiankang Deng et al.},
  year={2020},
  note={https://github.com/deepinsight/insightface}
}
```

---

## 10. License
This project is developed for educational and research purposes. Please refer to the original model licenses (DeepFace, InsightFace) for commercial use.
