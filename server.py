import cv2
import numpy as np
import threading
import time
import asyncio
from fastapi import FastAPI
from keras.models import load_model
from sse_starlette.sse import EventSourceResponse

# middleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# give access to localhost 5173 (Vite default port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and face detector
model = load_model('model_file_30epochs.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Shared state
latest_prediction = -1
FRAME_INTERVAL = 1.0 / 10  # 10 FPS

# Start webcam
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
            latest_prediction = -1
        else:
            for x, y, w, h in faces:
                face_img = gray[y:y+h, x:x+w]
                resized = cv2.resize(face_img, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                result = model.predict(reshaped, verbose=0)
                label = int(np.argmax(result, axis=1)[0])
                latest_prediction = label
                break

        elapsed = time.time() - start_time
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)

# Start background capture thread
threading.Thread(target=capture_emotion_loop, daemon=True).start()

from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse

@app.get("/stream_status")
async def stream_status():
    async def event_generator():
        global latest_prediction
        prev_value = None
        while True:
            if latest_prediction != prev_value:
                yield f"{latest_prediction}\n\n"
                prev_value = latest_prediction
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


# Clean up
import atexit
@atexit.register
def cleanup():
    cap.release()
    cv2.destroyAllWindows()


# uvicorn server:app --reload  
# 

@app.get("/test")
async def say_hello():
    return {"test": "Fireeeeee"}