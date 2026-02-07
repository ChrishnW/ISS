import numpy as np
from db.mysql import load_embeddings

EMBEDDINGS = load_embeddings()  # load once

THRESHOLD = 0.6

def match_face(embedding):
    best_score = 0
    best_id = None

    for person_id, ref_emb in EMBEDDINGS.items():
        score = np.dot(embedding, ref_emb)

        if score > best_score:
            best_score = score
            best_id = person_id

    if best_score > THRESHOLD:
        return best_id, best_score

    return None, None
