# display.py
import cv2

def draw_faces(frame, faces):
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def show_frame(frame, window_name="Live Face Detection"):
    cv2.imshow(window_name, frame)
