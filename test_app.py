import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import gradio as gr 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model('asl_model_tuned.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Label encoder to convert numerical labels back to original labels
label_encoder = LabelEncoder()
label_encoder.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'])

# Function to preprocess hand landmarks for prediction
def preprocess_landmarks(landmarks, fixed_length=63):
    data_aux = []
    for hand_landmarks in landmarks:
        for landmark in hand_landmarks.landmark:
            data_aux.extend([landmark.x, landmark.y, landmark.z])
    
    # Pad or trim the sequence to the fixed length
    if len(data_aux) > fixed_length * 3:
        data_aux = data_aux[:fixed_length * 3]
    data_padded = pad_sequences([data_aux], maxlen=fixed_length * 3, padding='post', dtype='float32')[0]

    # Reshape to the correct shape for the model
    reshaped_data = np.array(data_padded).reshape((1, fixed_length, 3, 1))

    return reshaped_data

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        # Preprocess landmarks for prediction
        landmarks = preprocess_landmarks(results.multi_hand_landmarks, fixed_length=63)
        
        # Predict the sign
        prediction = model.predict(landmarks)
        class_idx = np.argmax(prediction)
        class_label = label_encoder.inverse_transform([class_idx])[0]

        # Draw the landmarks and label on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the label
        cv2.putText(frame, class_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('ASL Sign Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Function to predict the sign from an image frame
def predict_from_frame(frame):
    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Preprocess landmarks for prediction
        landmarks = preprocess_landmarks(results.multi_hand_landmarks, fixed_length=63)
        
        # Predict the sign
        prediction = model.predict(landmarks)
        class_idx = np.argmax(prediction)
        class_label = label_encoder.inverse_transform([class_idx])[0]
        
        return class_label
    
    return 'No hand detected'

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_from_frame, 
    inputs=gr.Image(source="webcam", tool="editor", type="numpy"),
    outputs="text",
    title="ASL Sign Detection",
    description="Translate American Sign Language (ASL) signs into text."   
) 

# Launch the Gradio app
iface.launch()