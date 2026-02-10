# rtsp_reader.py
import cv2
import threading
import time

class RTSPReader:
    def __init__(self, url):
        self.url = url
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()

    def _reader(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("❌ Failed to open RTSP stream")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.lock:
                self.latest_frame = frame

        cap.release()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
