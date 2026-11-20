"""Reusable frame-level inference for the Streamlit webcam application."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

from utils.eye_aspect_ratio import eye_aspect_ratio
from utils.landmark_utils import crop_eye, get_left_eye, get_right_eye, shape_to_coords


class DrowsinessEngine:
    """Load expensive model assets once and create isolated per-camera sessions."""

    def __init__(self, assets: dict[str, str | Path]) -> None:
        required = {"eye_cnn", "mouth_cnn", "ear_classifier", "landmark_predictor"}
        missing = required.difference(assets)
        if missing:
            raise ValueError(f"Missing model asset keys: {', '.join(sorted(missing))}")

        self.eye_cnn = load_model(str(assets["eye_cnn"]), compile=False)
        self.mouth_cnn = load_model(str(assets["mouth_cnn"]), compile=False)

        import pickle
        with Path(assets["ear_classifier"]).open("rb") as model_file:
            self.ear_classifier = pickle.load(model_file)

        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(str(assets["landmark_predictor"]))
        # Model inference is serialized, while temporal state remains per visitor.
        self._inference_lock = threading.Lock()

    def create_session(self) -> "DrowsinessSession":
        """Create independent temporal counters for one browser camera stream."""
        return DrowsinessSession(self)


class DrowsinessSession:
    """Frame processor state belonging to exactly one browser camera stream."""

    EAR_THRESHOLD = 0.23
    EAR_CONSEC_FRAMES = 20
    EYE_CLOSED_THRESHOLD = 0.50
    EYE_CNN_CONSEC = 5
    YAWN_THRESHOLD = 0.50
    YAWN_CONSEC = 3
    SIDE_LOOK_SECONDS = 3.0
    SIDE_LEFT_RATIO = 0.35
    SIDE_RIGHT_RATIO = 0.65
    ML_CLOSED_CONSEC = 10

    def __init__(self, engine: DrowsinessEngine) -> None:
        self.engine = engine
        self.eye_cnn = engine.eye_cnn
        self.mouth_cnn = engine.mouth_cnn
        self.ear_classifier = engine.ear_classifier
        self.face_detector = engine.face_detector
        self.landmark_predictor = engine.landmark_predictor
        self._inference_lock = engine._inference_lock
        self._reset_temporal_state()

    def _reset_temporal_state(self) -> None:
        self.ear_counter = 0
        self.eye_cnn_counter = 0
        self.yawn_counter = 0
        self.ml_closed_counter = 0
        self.side_start_time: float | None = None

    @staticmethod
    def _model_input(crop: np.ndarray, model: Any) -> np.ndarray:
        """Preprocess a crop to match the saved model's channel dimensions."""
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = int(input_shape[1] or 64)
        width = int(input_shape[2] or 64)
        channels = int(input_shape[3] or 3)

        if crop is None or crop.size == 0:
            raise ValueError("The detected eye or mouth crop is empty.")

        if channels == 1:
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (width, height))
            image = np.expand_dims(image, axis=-1)
        elif channels == 3:
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
        else:
            raise ValueError(f"Unsupported model input channel count: {channels}")

        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    @staticmethod
    def _probability(model: Any, batch: np.ndarray) -> float:
        prediction = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
        if prediction.size == 0:
            raise ValueError("The neural network returned an empty prediction.")
        return float(np.clip(prediction[0], 0.0, 1.0))

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Return an annotated BGR frame and metrics for one incoming frame."""
        with self._inference_lock:
            return self._process_frame_locked(frame)

    def _process_frame_locked(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty video frame.")

        # Limit inference resolution to keep hosted CPU processing more responsive.
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640.0 / width
            frame = cv2.resize(
                frame, (640, int(height * scale)), interpolation=cv2.INTER_AREA
            )

        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 0)

        if not faces:
            self._reset_temporal_state()
            cv2.putText(
                annotated, "No face detected - look toward the camera",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 190, 255), 2, cv2.LINE_AA
            )
            return annotated, {
                "face_detected": False,
                "alert": False,
                "alert_message": "",
            }

        # Select the largest face so signals from different people are not mixed.
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self.landmark_predictor(gray, face)
        coordinates = shape_to_coords(shape)

        left_eye = get_left_eye(coordinates)
        right_eye = get_right_eye(coordinates)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # The Random Forest was trained on [left EAR, right EAR, mean EAR].
        features = np.asarray([[left_ear, right_ear, ear]], dtype=np.float32)
        ml_prediction = int(self.ear_classifier.predict(features)[0])
        ml_closed = ml_prediction == 1

        eye_crop = crop_eye(frame, list(left_eye) + list(right_eye))
        eye_open_probability = self._probability(
            self.eye_cnn, self._model_input(eye_crop, self.eye_cnn)
        )
        # flow_from_directory sorts closed_eyes before open_eyes:
        # class 0=closed, class 1=open. The sigmoid therefore estimates P(open).
        eye_closed_probability = 1.0 - eye_open_probability
        eye_closed = eye_closed_probability >= self.EYE_CLOSED_THRESHOLD

        mouth_crop = crop_eye(frame, coordinates[48:68])
        yawn_probability = self._probability(
            self.mouth_cnn, self._model_input(mouth_crop, self.mouth_cnn)
        )
        is_yawning = yawn_probability >= self.YAWN_THRESHOLD

        x_values = [point[0] for point in coordinates]
        min_x, max_x = min(x_values), max(x_values)
        nose_x = coordinates[30][0]
        face_width = max(max_x - min_x, 1)
        nose_position = (nose_x - min_x) / face_width
        side_looking = (
            nose_position < self.SIDE_LEFT_RATIO
            or nose_position > self.SIDE_RIGHT_RATIO
        )

        if side_looking:
            if self.side_start_time is None:
                self.side_start_time = time.monotonic()
            side_duration = time.monotonic() - self.side_start_time
        else:
            self.side_start_time = None
            side_duration = 0.0

        self.ear_counter = self.ear_counter + 1 if ear < self.EAR_THRESHOLD else 0
        self.eye_cnn_counter = self.eye_cnn_counter + 1 if eye_closed else 0
        self.yawn_counter = self.yawn_counter + 1 if is_yawning else 0
        self.ml_closed_counter = self.ml_closed_counter + 1 if ml_closed else 0

        ear_alert = self.ear_counter >= self.EAR_CONSEC_FRAMES
        eye_alert = self.eye_cnn_counter >= self.EYE_CNN_CONSEC
        yawn_alert = self.yawn_counter >= self.YAWN_CONSEC
        ml_alert = self.ml_closed_counter >= self.ML_CLOSED_CONSEC
        side_alert = side_duration >= self.SIDE_LOOK_SECONDS
        alert = ear_alert or eye_alert or yawn_alert or ml_alert or side_alert

        if side_alert:
            alert_message = "LOOK STRAIGHT ALERT"
        elif yawn_alert:
            alert_message = "YAWNING DETECTED"
        else:
            alert_message = "DROWSINESS ALERT" if alert else ""

        x1, y1 = max(0, face.left()), max(0, face.top())
        x2 = min(annotated.shape[1] - 1, face.right())
        y2 = min(annotated.shape[0] - 1, face.bottom())
        cv2.rectangle(
            annotated, (x1, y1), (x2, y2),
            (0, 0, 255) if alert else (0, 200, 100), 2
        )

        overlay_lines = [
            f"EAR: {ear:.3f} | EAR ML: {'CLOSED' if ml_closed else 'OPEN'}",
            f"Eye closed probability: {eye_closed_probability:.2f}",
            f"Yawn probability: {yawn_probability:.2f}",
            f"Head-turn duration: {side_duration:.1f}s",
        ]
        for index, line in enumerate(overlay_lines):
            cv2.putText(
                annotated, line, (14, 26 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (40, 230, 230), 2, cv2.LINE_AA
            )

        if alert:
            cv2.rectangle(
                annotated, (0, annotated.shape[0] - 48),
                (annotated.shape[1], annotated.shape[0]), (0, 0, 190), -1
            )
            cv2.putText(
                annotated, alert_message,
                (16, annotated.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        metrics = {
            "face_detected": True,
            "alert": alert,
            "alert_message": alert_message,
            "ear": ear,
            "eye_closed_probability": eye_closed_probability,
            "yawn_probability": yawn_probability,
            "ear_ml_status": "CLOSED" if ml_closed else "OPEN",
            "head_turn_seconds": side_duration,
        }
        return annotated, metrics
