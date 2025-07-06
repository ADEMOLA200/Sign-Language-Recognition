# performance_test.py
import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    max_num_hands=1
)

# Capture video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Latency measurements
latencies = []

# Run for 100 frames
for _ in range(100):
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe processing
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            x_.append(x)
            y_.append(y)

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))
        
        # Prediction
        prediction = model.predict([np.asarray(data_aux)])
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # in ms
    latencies.append(latency)

# Release resources
cap.release()
hands.close()

# Plot latency distribution
plt.figure(figsize=(10, 6))
plt.hist(latencies, bins=20, alpha=0.7, color='blue')
plt.title('End-to-End Latency Distribution per Frame')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('latency_distribution.png', dpi=300)
plt.close()

print(f"Average latency: {np.mean(latencies):.2f} ms")
print(f"Max latency: {max(latencies):.2f} ms")
print(f"Min latency: {min(latencies):.2f} ms")
