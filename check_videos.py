import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
videos = ["video.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]

print("=" * 60)
print("IBVAP VIDEO FILES DIAGNOSTIC CHECK")
print("=" * 60)

durations = {}
for vid in videos:
    path = os.path.join(BASE_DIR, vid)
    if not os.path.exists(path):
        print(f"[-] {vid}: FILE MISSING at {path}")
        continue
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[-] {vid}: FAILED TO OPEN WITH OPENCV")
        continue
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = frames / fps if fps else 0
    durations[vid] = dur
    ret, test_frame = cap.read()
    status = "OK (Readable)" if (ret and test_frame is not None) else "READ ERROR"
    print(f"[+] {vid:10s} | {frames:5.0f} frames | {fps:5.2f} FPS | {dur:5.2f}s | {w}x{h} | {status}")
    cap.release()

print("=" * 60)
if durations:
    min_vid = min(durations, key=durations.get)
    max_vid = max(durations, key=durations.get)
    print(f"Shortest video: {min_vid} ({durations[min_vid]:.2f}s)")
    print(f"Longest video:  {max_vid} ({durations[max_vid]:.2f}s)")
print("=" * 60)
