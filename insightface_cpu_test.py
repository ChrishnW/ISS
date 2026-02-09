import onnxruntime as ort
print("ORT device:", ort.get_device())

import insightface
app = insightface.app.FaceAnalysis(
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=-1)

print("InsightFace running on CPU successfully")
