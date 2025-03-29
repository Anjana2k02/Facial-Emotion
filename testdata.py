import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
model = load_model('model_file_30epochs.h5')

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Read the input image
frame = cv2.imread("faces-smallus.jpg")
if frame is None:
    print("Error: Image not found or unable to load.")
    exit()

# Convert image to grayscale (needed for face detection)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

# Loop through detected faces and classify emotions
for x, y, w, h in faces:
    # Extract the face region
    sub_face_img = gray[y:y+h, x:x+w]
    
    # Resize to 48x48 (model input size)
    resized = cv2.resize(sub_face_img, (48, 48))
    
    # Normalize pixel values to [0,1]
    normalized = resized / 255.0
    
    # Reshape to match model input dimensions (1, 48, 48, 1)
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    
    # Predict the emotion
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    
    # Draw rectangles around detected faces
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
    
    # Put the emotion label on the image
    cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display the output image
cv2.imshow("Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
