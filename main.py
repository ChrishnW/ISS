import os
import cv2
import insightface
import numpy as np
from datetime import datetime
import time

# ---------------- CONFIG ----------------
VIDEO_PATH = 'assets/videos/video5.mp4'   # replace with 0 for webcam or RTSP
SAMPLES_DIR = 'assets/images'             # reference images folder-per-person
RESULTS_DIR = 'results'
DETECT_WIDTH = 400                         # smaller width for CPU detection
MATCH_THRESHOLD = 0.6                      # threshold for sigmoid function
COOLDOWN = 3.0                             # seconds before logging same person
SHOW_VIDEO = True                           # display video window
TARGET_FPS = 30                             # target FPS for live feed
TOP_K = 3                                   # top candidate matches per face
DEBUG = False                                # print debug scores
COSINE_SIM_THRESHOLD = 0.25                # cosine similarity threshold (more lenient for video)
EUCLIDEAN_DIST_THRESHOLD = 0.65            # euclidean distance threshold (more lenient)
MIN_MATCH_CONF = 0.30                      # minimum confidence to display a match
CONFIRMED_MATCH_CONF = 0.40                # confidence needed to confirm match (green box)
# ----------------------------------------

# --- SETUP RESULTS FILE ---
os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = os.path.join(RESULTS_DIR, f"{timestamp_str}.txt")

# --- LOAD INSIGHTFACE (CPU model) ---
app = insightface.app.FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)

# --- LOAD REFERENCE EMBEDDINGS ---
person_embeddings = {}  # {person_name: average_embedding}
person_embeddings_list = {}  # {person_name: [embedding1, embedding2, ...]} for reference
print("[*] Loading reference embeddings from sample images...")
for person_name in os.listdir(SAMPLES_DIR):
    person_path = os.path.join(SAMPLES_DIR, person_name)
    if not os.path.isdir(person_path):
        continue
    embeddings = []
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = app.get(img)
        for face in faces:
            embeddings.append(face.embedding)
    if embeddings:
        # Store both individual embeddings and averaged embedding
        person_embeddings_list[person_name] = embeddings
        # Average embeddings for more stable matching
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # normalize
        person_embeddings[person_name] = avg_embedding
        print(f"  ✓ {person_name}: {len(embeddings)} face image(s) loaded")

print(f"\n[*] Loaded {len(person_embeddings)} people. Ready to process video...\n")

last_seen = {name: -COOLDOWN for name in person_embeddings}

# --- OPEN VIDEO OR CAMERA ---
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Cannot read video / camera")
    exit()

orig_h, orig_w = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else TARGET_FPS

# --- SETUP VIDEO WRITER ---
output_path = os.path.join(RESULTS_DIR, f"output_{timestamp_str}.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

frame_count = 0
frame_skip = 1

# --- DISTANCE & SIMILARITY FUNCTIONS ---
def euclidean_distance(emb1, emb2):
    """L2 euclidean distance"""
    return np.linalg.norm(emb1 - emb2)

def cosine_similarity(emb1, emb2):
    """Cosine similarity (0 to 1, higher is better)"""
    # Normalize embeddings
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return np.clip(np.dot(emb1_norm, emb2_norm), -1.0, 1.0)

def sigmoid_similarity(distance, threshold=0.5, steepness=10):
    """Convert embedding distance to 0-1 probability using sigmoid"""
    return 1 / (1 + np.exp(steepness * (distance - threshold)))

def compute_face_confidence(test_emb, person_avg_emb, person_embeddings_list):
    """
    Robust confidence computation for video frames with movement/distance variations.
    Uses multiple metrics weighted appropriately for real-world scenarios.
    """
    # Compare against ALL individual reference embeddings (not just average)
    distances = [euclidean_distance(test_emb, ref_emb) for ref_emb in person_embeddings_list]
    cosine_sims = [cosine_similarity(test_emb, ref_emb) for ref_emb in person_embeddings_list]

    # Use best match (closest reference)
    best_eucl_dist = min(distances)
    best_cosine_sim = max(cosine_sims)
    avg_eucl_dist = np.mean(distances)
    median_eucl_dist = np.median(distances)

    # Compute multiple scores
    # Score 1: Best Euclidean match (most reliable for video)
    eucl_score = sigmoid_similarity(best_eucl_dist, threshold=EUCLIDEAN_DIST_THRESHOLD, steepness=8)

    # Score 2: Average Euclidean (consensus across references)
    avg_eucl_score = sigmoid_similarity(avg_eucl_dist, threshold=EUCLIDEAN_DIST_THRESHOLD + 0.15, steepness=6)

    # Score 3: Median Euclidean (robust to outliers)
    median_eucl_score = sigmoid_similarity(median_eucl_dist, threshold=EUCLIDEAN_DIST_THRESHOLD + 0.05, steepness=7)

    # Score 4: Best Cosine similarity
    cosine_score = (best_cosine_sim + 1.0) / 2.0  # convert [-1, 1] to [0, 1]

    # Score 5: Distance-based confidence (how consistently close across all refs)
    min_dist = min(distances)
    max_dist = max(distances)
    consistency_score = 1.0 - min(0.3, (max_dist - min_dist) / 2.0)  # penalize high variance

    # Weighted combination optimized for video matching
    # Euclidean distance is most reliable indicator for video frames
    combined_prob = (
        eucl_score * 0.45 +          # best match distance
        median_eucl_score * 0.25 +   # robust average
        cosine_score * 0.20 +        # secondary metric
        consistency_score * 0.10     # consistency bonus
    )

    if DEBUG:
        print(f"    Eucl: {best_eucl_dist:.3f} ({eucl_score:.2f}) | "
              f"Cosine: {best_cosine_sim:.3f} ({cosine_score:.2f}) | "
              f"Combined: {combined_prob:.2f}")

    return combined_prob

with open(results_file, 'w') as f:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_display = frame.copy()
        scale = DETECT_WIDTH / frame.shape[1]
        frame_small = cv2.resize(frame, (DETECT_WIDTH, int(frame.shape[0]*scale)))

        start_time = time.time()
        faces = app.get(frame_small)
        for face in faces:
            emb = face.embedding
            emb_norm = emb / np.linalg.norm(emb)  # normalize for consistency

            # --- Compute confidence for each person ---
            candidate_probs = {}
            for person_name, avg_emb in person_embeddings.items():
                confidence = compute_face_confidence(
                    emb_norm,
                    avg_emb,
                    person_embeddings_list[person_name]
                )
                candidate_probs[person_name] = confidence

            # Sort ALL candidates by confidence
            sorted_candidates = sorted(candidate_probs.items(), key=lambda x: x[1], reverse=True)

            # Include all candidates as possible matches (lenient filtering)
            top_candidates = sorted_candidates[:TOP_K]

            # Determine if confident match exists
            match_exists = any(prob >= CONFIRMED_MATCH_CONF for _, prob in top_candidates)

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            match_line = f"[{timestamp:.2f}s] Found Face - Top Match: "

            if top_candidates:
                best_name, best_prob = top_candidates[0]
                match_line += f"{best_name} ({best_prob*100:.1f}%)"
                f.write(match_line + "\n")
                print(match_line)

                # Show all top candidates if above minimum threshold
                if best_prob >= MIN_MATCH_CONF:
                    for name, prob in top_candidates:
                        if prob >= MIN_MATCH_CONF:
                            line = f"  → {name}: {prob*100:.1f}%"
                            f.write(line + "\n")
                            print(line)
            else:
                f.write(match_line + "UNKNOWN\n")
                print(match_line + "UNKNOWN")

            # Draw bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            color = (0, 255, 0) if match_exists else (0, 165, 255)  # green if confirmed, orange if possible
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)

            # Draw candidate names
            if top_candidates:
                best_name, best_prob = top_candidates[0]
                text_color = (0, 255, 0) if match_exists else (0, 165, 255)
                cv2.putText(frame_display, f"{best_name} ({best_prob*100:.0f}%)",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        out.write(frame_display)

        if SHOW_VIDEO:
            cv2.imshow("Face Monitoring", frame_display)
            key = cv2.waitKey(1)
            if key == 27:
                break

        # --- DYNAMIC FRAME SKIP ---
        elapsed = time.time() - start_time
        target_time = 1.0 / TARGET_FPS
        if elapsed > target_time:
            frame_skip = min(frame_skip + 1, 5)
        else:
            frame_skip = max(frame_skip - 1, 1)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nResults saved to {results_file}")
print(f"Video output saved to {output_path}")
