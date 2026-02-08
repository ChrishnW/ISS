import onnxruntime as ort
print("ONNX device:", ort.get_device())

import insightface
app = insightface.app.FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0)

print("InsightFace providers:", app.providers)
