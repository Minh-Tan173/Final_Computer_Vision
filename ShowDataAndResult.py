import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Face Embedding Analysis", layout="wide")

# === Utility ===
def load_db(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def cosine_sim(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Load both databases ===
db_vgg = load_db("db_VGG-Face.pkl")
db_mobile = load_db("db_MobileFaceNet_Insight.pkl")

# === SECTION 1: Introduction ===
st.title("📊 Face Embedding Analysis & Model Comparison")
st.markdown("""
### 1. Introduction
This dashboard presents a comparative evaluation of two face recognition models: **VGG-Face** and **MobileFaceNet (InsightFace)**.
We analyze the extracted embeddings and their effectiveness in distinguishing between different individuals.
""")

# === SECTION 2: Dataset Summary ===
st.markdown("### 2. Dataset Summary")
num_images = len(db_vgg)
all_names = sorted(set([v['name'] for v in db_vgg.values()]))
st.write(f"- Total Images: `{num_images}`")
st.write(f"- Unique Persons: `{len(all_names)}`")
st.write(f"- Names: {', '.join(all_names)}")

# === SECTION 3: Embedding Viewer ===
st.markdown("### 3. Feature Embedding Visualization")
selected_name = st.selectbox("Select a person to view samples", all_names)

cols = st.columns(3)
count = 0
for path, info in db_vgg.items():
    if info["name"] == selected_name:
        img = Image.open(path) if os.path.exists(path) else None
        if img:
            with cols[count % 3]:
                st.image(img, caption=os.path.basename(path), use_column_width=True)
        count += 1

# === SECTION 4: Pairwise Similarity (Single Model) ===
st.markdown("### 4. Pairwise Similarity Analysis")
selected_model = st.radio("Select model to analyze", ["VGG-Face", "MobileFaceNet"])
db = db_vgg if selected_model == "VGG-Face" else db_mobile

paths = list(db.keys())
id_labels = [f"{db[p]['name']}_{os.path.basename(p)}" for p in paths]

sim_matrix = np.zeros((len(paths), len(paths)))
for i in range(len(paths)):
    for j in range(len(paths)):
        sim_matrix[i][j] = cosine_sim(db[paths[i]]['embedding'], db[paths[j]]['embedding'])

df_sim = pd.DataFrame(sim_matrix, index=id_labels, columns=id_labels)
st.dataframe(df_sim.style.background_gradient(cmap="YlGnBu"))

# === SECTION 5: Model Comparison ===
st.markdown("### 5. Comparative Evaluation of Models")

common_paths = list(set(db_vgg.keys()) & set(db_mobile.keys()))
common_paths.sort()

results = []
for i in range(len(common_paths)):
    for j in range(i + 1, len(common_paths)):
        path_i, path_j = common_paths[i], common_paths[j]
        name1, name2 = db_vgg[path_i]["name"], db_vgg[path_j]["name"]

        sim_vgg = cosine_sim(db_vgg[path_i]["embedding"], db_vgg[path_j]["embedding"])
        sim_mobile = cosine_sim(db_mobile[path_i]["embedding"], db_mobile[path_j]["embedding"])
        label = "Same" if name1 == name2 else "Different"

        better = "VGG-Face" if abs(sim_vgg - (1 if label == "Same" else 0)) < abs(sim_mobile - (1 if label == "Same" else 0)) else "MobileFaceNet"
        comment = "✅ Both good"
        if label == "Same" and (sim_vgg < 0.85 or sim_mobile < 0.85):
            comment = "⚠️ Low similarity for same person"
        elif label == "Different" and (sim_vgg > 0.75 or sim_mobile > 0.75):
            comment = "❗ High similarity for different persons"

        results.append({
            "Image A": os.path.basename(path_i),
            "Image B": os.path.basename(path_j),
            "Label": label,
            "Cosine VGG": sim_vgg,
            "Cosine Mobile": sim_mobile,
            "Better Model": better,
            "Comment": comment
        })

df_compare = pd.DataFrame(results)
st.dataframe(df_compare.style.highlight_max(axis=1, subset=["Cosine VGG", "Cosine Mobile"]))


# === SECTION 6: Conclusion ===
st.markdown("### 6. Conclusion")
total = len(df_compare)
vgg_better = (df_compare["Better Model"] == "VGG-Face").sum()
mobile_better = (df_compare["Better Model"] == "MobileFaceNet").sum()

st.markdown(f"""
- 🧠 **VGG-Face performed better on {vgg_better}/{total} comparisons**
- ⚡ **MobileFaceNet performed better on {mobile_better}/{total} comparisons**

> 🔍 This analysis shows that **VGG-Face** tends to perform better on identity consistency,
while **MobileFaceNet** is efficient and suitable for real-time systems. Further improvement may be achieved via preprocessing (face alignment, lighting normalization).
""")
