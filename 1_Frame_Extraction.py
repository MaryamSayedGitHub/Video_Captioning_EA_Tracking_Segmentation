import cv2


def extract_frames(video_path, fps_sample=1):
    """
    Splits the video into frames.
    Returns a list of tuples: (timestamp in seconds, frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps / fps_sample))

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            timestamp = count / video_fps
            frames.append((timestamp, frame))
        count += 1

    cap.release()
    print(f"[Frame Extraction] Extracted {len(frames)} frames from {video_path}")
    return frames