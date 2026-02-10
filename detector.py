# detector.py
import insightface
import cv2

class FaceDetector:
    def __init__(self, model_name, providers, detect_width=1080):
        self.app = insightface.app.FaceAnalysis(
            name=model_name,
            providers=providers
        )
        self.app.prepare(ctx_id=-1)
        self.detect_width = detect_width

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = self.detect_width / w
        small = cv2.resize(frame, (self.detect_width, int(h * scale)))
        faces = self.app.get(small)

        results = []
        for face in faces:
            # Scale back to original frame size
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            results.append({
                "bbox": (x1, y1, x2, y2),
                "face_obj": face
            })
        return results
