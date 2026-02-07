import cv2
import time
from recognition.face_detector import detect_faces
from recognition.face_encoder import encode_face
from recognition.matcher import match_face
from utils.cooldown import Cooldown
from db.mysql import save_detection

CAMERA_FPS = 3

class CameraWorker:
    def __init__(self, camera_id, rtsp_url):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.cooldown = Cooldown(seconds=45)

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)

        if not cap.isOpened():
            raise RuntimeError("Cannot open RTSP stream")

        last_frame_time = 0

        while True:
            now = time.time()

            if now - last_frame_time < 1 / CAMERA_FPS:
                continue

            last_frame_time = now
            ret, frame = cap.read()

            if not ret:
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                continue

            faces = detect_faces(frame)

            for face_img in faces:
                embedding = encode_face(face_img)
                person_id, score = match_face(embedding)

                if person_id and self.cooldown.allowed(person_id):
                    save_detection(
                        person_id=person_id,
                        camera_id=self.camera_id
                    )

