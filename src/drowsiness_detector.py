import cv2
import os
import sys
import dlib
import time
import pickle
import platform
import threading
import numpy as np
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow.keras.models import load_model
from utils.landmark_utils import shape_to_coords, get_left_eye, get_right_eye, crop_eye
from utils.eye_aspect_ratio import eye_aspect_ratio

try:
    from playsound import playsound
except Exception:
    playsound = None

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
PREDICTOR_PATH = os.path.join(PROJECT_ROOT, 'shape_predictor_68_face_landmarks.dat')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)
ALERT_SOUND_PATH = os.path.join(ASSETS_DIR, 'siren-alert.mp3')
ALERT_AUDIO_COOLDOWN = 3.0

EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 20
EYE_CNN_CLOSE_THRESH = 0.5
EYE_CNN_CONSEC = 5
YAWN_THRESH = 0.6
YAWN_CONSEC = 3
SIDE_LOOK_THRESH = 3.0
SIDE_LEFT_RATIO = 0.35
SIDE_RIGHT_RATIO = 0.65
SAVE_DEBUG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'debug_mouth')
os.makedirs(SAVE_DEBUG_DIR, exist_ok=True)
ALERT_LOG = os.path.join(PROJECT_ROOT, 'outputs', 'alerts.log')
SAVE_DEBUG_THRESH = 0.4

eye_cnn = load_model(os.path.join(MODEL_DIR, 'cnn_model.h5'))
mouth_cnn = load_model(os.path.join(MODEL_DIR, 'mouth_cnn_model.h5'))
with open(os.path.join(MODEL_DIR, 'ml_model.pkl'), 'rb') as f:
    ear_model = pickle.load(f)

detector = dlib.get_frontal_face_detector()
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(
        f"Dlib predictor file not found at {PREDICTOR_PATH}.\n"
        "Please download `shape_predictor_68_face_landmarks.dat` from http://dlib.net/files/"
        " and place it in the project root or update PREDICTOR_PATH."
    )
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv2.VideoCapture(0)

ear_counter = 0
eye_cnn_counter = 0
yawn_counter = 0
side_start_time = None
last_audio_alert = 0.0


def play_audio_alert():
    """Play the configured alert sound without blocking the main loop."""
    def _play():
        if playsound and os.path.exists(ALERT_SOUND_PATH):
            try:
                playsound(ALERT_SOUND_PATH)
                return
            except Exception:
                pass
        if platform.system() == "Windows":
            try:
                import winsound
                winsound.Beep(2500, 700)
            except Exception:
                pass

    threading.Thread(target=_play, daemon=True).start()

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    norm = resized.astype('float32') / 255.0
    norm = np.expand_dims(norm, axis=(0, -1))
    return norm

def crop_region(frame, pts):
    x = pts[:,0]; y = pts[:,1]
    x1, x2 = np.min(x), np.max(x)
    y1, y2 = np.min(y), np.max(y)
    return frame[y1:y2, x1:x2]

print("Realtime detector running... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)

    for rect in rects:
        shape = predictor(gray, rect)
        coords = shape_to_coords(shape)
        left_eye = np.array(get_left_eye(coords))
        right_eye = np.array(get_right_eye(coords))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        try:
            ml_pred = ear_model.predict([[ear]])[0]
        except Exception:
            ml_pred = None

        try:
            eye_region = crop_eye(frame, list(left_eye) + list(right_eye))
        except Exception:
            eye_region = frame
        try:
            eye_input = preprocess(eye_region)
            eye_prob = eye_cnn.predict(eye_input)[0][0]
        except Exception:
            eye_prob = 0.0
        eye_closed = eye_prob >= EYE_CNN_CLOSE_THRESH

        mouth = coords[48:68]
        try:
            mouth_region = crop_eye(frame, mouth)
        except Exception:
            mouth_region = frame
        try:
            mouth_input = preprocess(mouth_region)
            yawn_prob = mouth_cnn.predict(mouth_input)[0][0]
        except Exception:
            yawn_prob = 0.0
        is_yawning = yawn_prob >= YAWN_THRESH

        try:
            if yawn_prob >= SAVE_DEBUG_THRESH:
                import datetime
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                fname = os.path.join(SAVE_DEBUG_DIR, f'mouth_{ts}.jpg')
                try:
                    cv2.imwrite(fname, mouth_region)
                except Exception:
                    pass
                try:
                    with open(ALERT_LOG, 'a', encoding='utf-8') as lf:
                        lf.write(f"{ts}, MOUTH_CROP_SAVED, yawn_prob={yawn_prob:.3f}\n")
                except Exception:
                    pass
        except Exception:
            pass

        x_coords = [p[0] for p in coords]
        minx, maxx = min(x_coords), max(x_coords)
        nose_x = coords[30][0]
        face_w = maxx - minx if (maxx - minx) > 0 else 1
        nose_norm = (nose_x - minx) / face_w
        side_looking = (nose_norm < SIDE_LEFT_RATIO) or (nose_norm > SIDE_RIGHT_RATIO)
        if side_looking:
            if side_start_time is None:
                side_start_time = time.time()
            side_duration = time.time() - side_start_time
        else:
            side_start_time = None
            side_duration = 0.0

        if ear < EAR_THRESHOLD:
            ear_counter += 1
        else:
            ear_counter = 0

        if eye_closed:
            eye_cnn_counter += 1
        else:
            eye_cnn_counter = 0

        if is_yawning:
            yawn_counter += 1
        else:
            yawn_counter = 0

        if (ear_counter >= EAR_CONSEC_FRAMES or
            eye_cnn_counter >= EYE_CNN_CONSEC or
            yawn_counter >= YAWN_CONSEC or
            side_duration >= SIDE_LOOK_THRESH):

            if side_duration >= SIDE_LOOK_THRESH:
                msg = "LOOK STRAIGHT ALERT!"
            elif yawn_counter >= YAWN_CONSEC:
                msg = "DRINK WATER ALERT - Stop driving and drink water"
            else:
                msg = "DROWSINESS ALERT!"

            now = time.time()
            if now - last_audio_alert >= ALERT_AUDIO_COOLDOWN:
                play_audio_alert()
                last_audio_alert = now

            cv2.putText(frame, msg, (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
            cv2.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]),
                          (0,0,255), 4)

        cv2.putText(frame, f"EAR: {ear:.2f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Eye Prob: {eye_prob:.2f}", (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Yawn Prob: {yawn_prob:.2f}", (20,140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"NosePos: {nose_norm:.2f}", (20,170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
        if side_duration > 0:
            cv2.putText(frame, f"SideSecs: {side_duration:.1f}s", (20,200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
