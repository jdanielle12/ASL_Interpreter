import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('asl_model.keras')

# Define a function to predict a single frame
def predict_single_frame(frame, model, labels):
    # Preprocess the frame
    img_array = cv2.resize(frame, (64, 64))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = labels[predicted_class_index[0]]
    
    return predicted_label

# Set up ImageDataGenerator to get class labels
train_dir = 'ASL_Alphabet_Dataset/asl_alphabet_train'
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Get the class labels
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Predict the hand sign
        predicted_label = predict_single_frame(frame, model, labels)

        # Display the resulting frame with prediction
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hand Sign Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
