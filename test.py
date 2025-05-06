import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the trained model
# model = load_model('model_file_30epochs.h5')
model = load_model('FER_model.h5')

# Open webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load face detection cascade
face_cascade_path = 'haarcascade_frontalface_default.xml'
faceDetect = cv2.CascadeClassifier(face_cascade_path)

if faceDetect.empty():
    print("Error: Haarcascade file not found!")
    exit()

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

while True:
    ret, frame = video.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    for x, y, w, h in faces:
        # Extract face region
        sub_face_img = gray[y:y+h, x:x+w]

        # Resize face to match model input
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0  # Normalize pixel values
        reshaped = np.reshape(normalize, (1, 48, 48, 1))  # Reshape for model input

        # Predict emotion
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw face rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the output
    cv2.imshow("Facial Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
