# import onnxruntime as ort
# print(ort.get_device())

import insightface
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)
print("InsightFace loaded with GPU")
