import cv2
import numpy as np
from tensorflow.keras.models import load_model
import winsound
import time

# Load model
model = load_model('model/best_model.keras')

IMG_SIZE = 224

# Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

cap = cv2.VideoCapture(0)

# Stability + Logic
closed_frames = 0
FRAME_THRESHOLD = 20   # ~2–3 sec

# Sound control
last_beep_time = 0
BEEP_COOLDOWN = 2

# Prediction smoothing
pred_buffer = []
BUFFER_SIZE = 7

# Face tracking (VERY IMPORTANT)
prev_face = None
alarm_active = False   # ✅ ADD THIS

def preprocess_eye(eye):
    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
    eye = eye / 255.0
    eye = np.expand_dims(eye, axis=0)
    return eye

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    status = "No Face"

    # Detect face
    if prev_face is None:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            prev_face = faces[0]
    else:
        (x, y, w, h) = prev_face
        faces = [(x, y, w, h)]

    face_detected = False  # ✅ NEW

    for (x, y, w, h) in faces:
        face_detected = True

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5)

        preds = []

        for (ex, ey, ew, eh) in eyes[:2]:
            eye = roi_color[ey:ey+eh, ex:ex+ew]

            eye_img = preprocess_eye(eye)
            pred = model.predict(eye_img, verbose=0)[0][0]

            preds.append(pred)

            cv2.rectangle(frame,
                          (x+ex, y+ey),
                          (x+ex+ew, y+ey+eh),
                          (255, 0, 0), 2)

        # Smooth prediction
        if len(preds) > 0:
            avg_pred = np.mean(preds)

            pred_buffer.append(avg_pred)
            if len(pred_buffer) > BUFFER_SIZE:
                pred_buffer.pop(0)

            smooth_pred = np.mean(pred_buffer)

            # Decision
            if smooth_pred < 0.5:
                closed_frames += 1
            else:
                closed_frames = 0
                alarm_active = False  # ✅ FIX

        else:
            # No eyes detected → reset
            closed_frames = 0
            alarm_active = False

        # ALERT LOGIC
        if closed_frames > FRAME_THRESHOLD:
            status = "🚨 SLEEPY 😴"

            cv2.putText(frame, "SLEEP ALERT!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

            # Beep only once
            if not alarm_active:
                winsound.Beep(2000, 800)
                alarm_active = True
        else:
            status = "AWAKE 👀"

        # Draw face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # If face lost → reset everything
    if not face_detected:
        prev_face = None
        closed_frames = 0
        alarm_active = False
        status = "No Face"

    # Display status
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()