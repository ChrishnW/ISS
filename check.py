import cv2
import insightface
import onnxruntime as ort

print("OpenCV:", cv2.__version__)
print("InsightFace:", insightface.__version__)
print("ONNX device:", ort.get_device())
