import cv2
import os
import numpy as np
import insightface
import onnxruntime as ort
from datetime import timedelta

# ======================
# SETTINGS
# ======================
VIDEO_PATH = "assets/videos/1.mp4"
KNOWN_FACES_DIR = "assets/images"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "matches.txt")

FRAME_SKIP = 10
MATCH_THRESHOLD = 0.6

# ======================
# PREP OUTPUT
# ======================
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================
# INIT MODEL
# ======================
print("ONNX device:", ort.get_device())

app = insightface.app.FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0)

# ======================
# LOAD KNOWN FACES
# ======================
known_embeddings = {}

print("Loading known faces...")
for file in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, file)
    img = cv2.imread(path)

    if img is None:
        continue

    faces = app.get(img)
    if len(faces) == 0:
        print(f"No face found in {file}")
        continue

    known_embeddings[file] = faces[0].normed_embedding
    print(f"Loaded: {file}")

# ======================
# VIDEO SCAN
# ======================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0
print("\nScanning video...\n")

with open(RESULTS_FILE, "a", encoding="utf-8") as f:
    f.write(f"\n=== Scan started for {VIDEO_PATH} ===\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    timestamp_sec = frame_count / fps
    faces = app.get(frame)

    for face in faces:
        emb = face.normed_embedding

        for name, ref_emb in known_embeddings.items():
            score = np.dot(emb, ref_emb)

            if score > MATCH_THRESHOLD:
                timestamp = str(timedelta(seconds=int(timestamp_sec)))

                line = (
                    f"[MATCH] {name} | "
                    f"time={timestamp} | "
                    f"score={score:.2f}\n"
                )

                with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                    f.write(line)

cap.release()

with open(RESULTS_FILE, "a", encoding="utf-8") as f:
    f.write("=== Scan finished ===\n")

print("Scan complete. Results saved to:", RESULTS_FILE)
