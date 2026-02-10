# main.py
from config import RTSP_URL, DETECT_WIDTH, MODEL_NAME, PROVIDERS
from rtsp_reader import RTSPReader
from detector import FaceDetector
from display import draw_faces, show_frame
import cv2

# ---------- INIT ----------
reader = RTSPReader(RTSP_URL)
reader.start()

detector = FaceDetector(MODEL_NAME, PROVIDERS, DETECT_WIDTH)

# ---------- MAIN LOOP ----------
running = True
while running:
    frame = reader.get_frame()
    if frame is None:
        continue

    faces = detector.detect(frame)
    frame_with_boxes = draw_faces(frame, faces)
    show_frame(frame_with_boxes)

    if cv2.waitKey(1) == 27:  # ESC
        running = False
        break

cv2.destroyAllWindows()
print("🛑 Face detection stopped")
