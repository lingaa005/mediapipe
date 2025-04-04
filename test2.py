import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Input and Output Video Paths
input_video_path = "videoplayback.mp4"
output_video_path = "output_annotated.mp4"

# Open Video File
cap = cv2.VideoCapture(input_video_path)

# Get Video Properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer to Save Output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process Each Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert Frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw Pose Landmarks if Detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,  
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3), 
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
        )

    # Write Annotated Frame to Video
    out.write(frame)

    # Display Frame (Optional)
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

# Release Resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
