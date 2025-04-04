import cv2
import os

video_path = 'videoplayback.mp4'
output_dir = 'frames'

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 5  # Extract 5 frames per second

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
        filename = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames to {output_dir}")
