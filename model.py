import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------
# Load CSV of correct poses
# ------------------------------
CSV_PATH = "pose_dataset.csv"  # Make sure this file exists
df = pd.read_csv(CSV_PATH).drop(columns=["frame"], errors="ignore")
X_correct = df.to_numpy(dtype=np.float32)

# ------------------------------
# Define Autoencoder model
# ------------------------------
class PoseAutoencoder(nn.Module):
    def __init__(self, input_size=99, latent_size=32):
        super(PoseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ------------------------------
# Train the model
# ------------------------------
model = PoseAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_tensor = torch.tensor(X_correct, dtype=torch.float32)

print("🚀 Training model on correct poses...")
for epoch in range(100):
    model.train()
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
print("✅ Training complete!")

# ------------------------------
# Pose Evaluation Functions
# ------------------------------
JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

def analyze_pose(pose_vector, threshold=0.05):
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(pose_vector, dtype=torch.float32).unsqueeze(0)
        recon = model(inp)
        error = (inp - recon).squeeze().numpy()

    joint_errors = []
    for i in range(33):
        x_err = error[i * 3]
        y_err = error[i * 3 + 1]
        v_err = error[i * 3 + 2]
        dist = np.sqrt(x_err**2 + y_err**2)
        if dist > threshold:
            joint_errors.append((i, dist))
    return joint_errors

def give_feedback(joint_errors):
    if not joint_errors:
        return "✅ Good posture!"
    feedback = []
    for idx, err in joint_errors:
        name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"joint_{idx}"
        feedback.append(f"⚠️ Adjust your {name} (error: {err:.2f})")
    return "\n".join(feedback)

# ------------------------------
# Analyze New Video
# ------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

VIDEO_PATH = "videoplayback.mp4"  # Your video to evaluate

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

print("\n🎥 Processing test video for pose feedback...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        pose_vector = []
        for lm in result.pose_landmarks.landmark:
            pose_vector.extend([lm.x, lm.y, lm.visibility])

        if len(pose_vector) == 99:
            joint_errors = analyze_pose(pose_vector)
            feedback = give_feedback(joint_errors)
            print(f"\n🖼 Frame {frame_idx}")
            print(feedback)

    frame_idx += 1

cap.release()
pose.close()

print("\n✅ All frames analyzed!")
