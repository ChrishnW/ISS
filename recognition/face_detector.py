import insightface

model = insightface.app.FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
model.prepare(ctx_id=0)

def detect_faces(frame):
    faces = model.get(frame)
    return [f.crop for f in faces]
