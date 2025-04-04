import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the annotated video
input_video_path = "output_annotated.mp4"  # Replace with your video path
cap = cv2.VideoCapture(input_video_path)

# List to store extracted pose data
data_list = []

frame_count = 0  # Track frame numbers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Extract landmarks if detected
    if results.pose_landmarks:
        frame_data = {"frame": frame_count}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            frame_data[f"x_{idx}"] = landmark.x  # Normalized X coordinate
            frame_data[f"y_{idx}"] = landmark.y  # Normalized Y coordinate
            frame_data[f"visibility_{idx}"] = landmark.visibility  # Visibility score

        data_list.append(frame_data)  # Add to dataset

    frame_count += 1  # Increment frame counter

# Save data to CSV
df = pd.DataFrame(data_list)
df.to_csv("pose_dataset.csv", index=False)

cap.release()

print("Dataset saved as pose_dataset.csv 🎯")
