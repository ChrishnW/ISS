import cv2

# Setup video writer
out_path = "assets/videos/video1.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_small.shape[1], frame_small.shape[0]))

# Inside loop
out.write(frame_small)

# After loop
out.release()
