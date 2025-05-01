import cv2
import numpy as np
import threading
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from keras.models import load_model
from typing import Generator
import asyncio 
from sse_starlette.sse import EventSourceResponse



app = FastAPI()

# Load model and face detector
model = load_model('model_file_30epochs.h5')
face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Shared variable for camera frame
latest_prediction = -1

# Frame rate control (e.g., 10 FPS)
FRAME_INTERVAL = 1.0 / 10  # 10 frames per second

# Threaded video capture
cap = cv2.VideoCapture(0)
def capture_emotion_loop():
    global latest_prediction
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            latest_prediction = -1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        if len(faces) == 0:
            latest_prediction = -1  # No face found
        else:
            for x, y, w, h in faces:
                face_img = gray[y:y+h, x:x+w]
                resized = cv2.resize(face_img, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                result = model.predict(reshaped, verbose=0)
                label = int(np.argmax(result, axis=1)[0])
                latest_prediction = label
                break  # Only process the first face

        elapsed = time.time() - start_time
        time_to_wait = FRAME_INTERVAL - elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)


# Start background thread
threading.Thread(target=capture_emotion_loop, daemon=True).start()

# Streaming generator
def prediction_stream() -> Generator[str, None, None]:
    async def gen():
        while True:
            yield f"data: {latest_prediction}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/stream_status")
async def stream_status():
    async def event_generator():
        prev_value = None
        while True:
            if latest_prediction != prev_value:
                yield f"data: {latest_prediction}\n\n"
                prev_value = latest_prediction
            await asyncio.sleep(0.5)
    return EventSourceResponse(event_generator())

# Graceful shutdown
import atexit
@atexit.register
def cleanup():
    cap.release()
    cv2.destroyAllWindows()
