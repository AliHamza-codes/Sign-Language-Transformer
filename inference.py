import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 1. Setup
actions = np.array(['hello', 'thanks', 'iloveyou'])
model = load_model('action.h5')
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Drawing function for confidence bars
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), (245, 117, 16), -1)
        cv2.putText(output_frame, f'{actions[num]}: {prob*100:.0f}%', (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return output_frame

sequence = []
threshold = 0.8 # Isse 0.9 karke mazeed sakht kar sakte hain

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        
        # 2. Prediction Logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Last 30 frames
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            # Visualization logic
            image = prob_viz(res, actions, image)
            
            # Show "Null" if no action is above threshold
            current_action = actions[np.argmax(res)] if res[np.argmax(res)] > threshold else "Null"
            
            cv2.rectangle(image, (0,0), (640, 40), (0, 0, 0), -1)
            cv2.putText(image, f'Result: {current_action}', (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign Language Transformer', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()