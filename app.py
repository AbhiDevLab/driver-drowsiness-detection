from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.asset_manager import resolve_model_assets


ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("driver_drowsiness_streamlit")

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Driver Drowsiness Detection")
st.caption(
    "Browser-based webcam monitoring using facial landmarks, EAR, CNN inference, "
    "and a Random Forest classifier."
)

with st.sidebar:
    st.header("Detection settings")
    st.markdown(
        """
        - **EAR alert:** below 0.23 for 20 frames
        - **Eye CNN:** closed probability at least 0.50 for 5 frames
        - **Yawn CNN:** probability at least 0.50 for 3 frames
        - **EAR Random Forest:** closed for 10 frames
        - **Head turn:** sustained for 3 seconds
        """
    )
    st.divider()
    st.caption(
        "This is an educational prototype, not a medically validated or "
        "vehicle-certified safety system."
    )

st.info(
    "Privacy: live camera frames are transmitted to this Streamlit server for "
    "processing. This app does not intentionally save or record the video. "
    "Use the camera only with the subject's consent."
)


def _get_secret(key: str) -> str | None:
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None
    return str(value).strip() if value else None


asset_paths, missing_assets, asset_errors = resolve_model_assets(ROOT, st.secrets)

if missing_assets or asset_errors:
    st.subheader("Model assets need to be configured")
    st.warning(
        "The GitHub repository does not contain the trained models or the "
        "dlib 68-point landmark predictor. The webcam detector cannot start "
        "until all four files are available."
    )

    if missing_assets:
        st.markdown("**Missing files**")
        for item in missing_assets:
            st.write(f"- {item}")

    if asset_errors:
        st.markdown("**Asset download errors**")
        for item in asset_errors:
            st.error(item)

    st.markdown(
        """
        **Configure the four direct HTTPS download URLs**

        In Streamlit Community Cloud, open your app's **Settings → Secrets** and
        add the TOML entries below. Replace each example with a direct download
        URL. The dlib predictor URL must point to the extracted DAT file, not
        the compressed BZ2 archive.
        """
    )
    st.code(
        'EYE_CNN_URL = "https://your-host/cnn_model.h5"\n'
        'MOUTH_CNN_URL = "https://your-host/mouth_cnn_model.h5"\n'
        'EAR_MODEL_URL = "https://your-host/ml_model.pkl"\n'
        'DLIB_PREDICTOR_URL = "https://your-host/shape_predictor_68_face_landmarks.dat"',
        language="toml",
    )
    st.caption(
        "For local testing, place the three model files in models/ and the "
        "predictor file in the repository root. Local files take precedence "
        "over secret URLs."
    )
    st.stop()

try:
    from src.detection_engine import DrowsinessEngine

    @st.cache_resource(show_spinner="Loading the drowsiness detection models...")
    def load_engine(asset_items: tuple[tuple[str, str], ...]) -> DrowsinessEngine:
        return DrowsinessEngine({key: path for key, path in asset_items})

    engine = load_engine(
        tuple((key, str(path)) for key, path in sorted(asset_paths.items()))
    )
except Exception as exc:
    LOGGER.exception("Could not initialize the detection engine.")
    st.error(
        "The model files were found, but the detection engine could not load "
        "them. Check that the files are valid and were produced by compatible "
        "TensorFlow/Keras and scikit-learn versions."
    )
    st.exception(exc)
    st.stop()


class DrowsinessVideoProcessor(VideoProcessorBase):
    def __init__(self, shared_engine: DrowsinessEngine) -> None:
        self.engine = shared_engine
        self._metrics_lock = threading.Lock()
        self._latest_metrics: dict[str, Any] = {
            "face_detected": False,
            "alert": False,
            "alert_message": "",
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        try:
            annotated, metrics = self.engine.process_frame(image)
            with self._metrics_lock:
                self._latest_metrics = metrics
        except Exception as exc:
            LOGGER.exception("Frame processing failed.")
            annotated = image.copy()
            with self._metrics_lock:
                self._latest_metrics = {
                    "face_detected": False,
                    "alert": False,
                    "alert_message": "",
                    "processing_error": f"{type(exc).__name__}: {exc}",
                }
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_latest_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return dict(self._latest_metrics)


rtc_configuration: dict[str, Any] = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}
turn_url = _get_secret("TURN_SERVER_URL")
turn_username = _get_secret("TURN_SERVER_USERNAME")
turn_credential = _get_secret("TURN_SERVER_CREDENTIAL")
if turn_url and turn_username and turn_credential:
    rtc_configuration["iceServers"].append(
        {
            "urls": [turn_url],
            "username": turn_username,
            "credential": turn_credential,
        }
    )

st.subheader("Live camera")
st.write(
    "Allow camera access in your browser, then start the video stream. "
    "Keep your face clearly visible and reasonably well lit."
)

webrtc_context = webrtc_streamer(
    key="driver-drowsiness-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: DrowsinessVideoProcessor(engine),
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=rtc_configuration,
    async_processing=True,
)

st.subheader("Live detection metrics")


@st.fragment(run_every="1s")
def render_live_metrics() -> None:
    processor = webrtc_context.video_processor
    if processor is None:
        st.info("Start the camera stream to see live detection metrics.")
        return

    metrics = processor.get_latest_metrics()
    if metrics.get("processing_error"):
        st.error(f"Frame processing error: {metrics['processing_error']}")
        return

    if not metrics.get("face_detected"):
        st.warning("No face detected. Position your face in the camera view.")
        return

    columns = st.columns(4)
    columns[0].metric("Eye Aspect Ratio", f"{metrics['ear']:.3f}")
    columns[1].metric(
        "Eye closed probability",
        f"{metrics['eye_closed_probability']:.0%}",
    )
    columns[2].metric("Yawn probability", f"{metrics['yawn_probability']:.0%}")
    columns[3].metric("EAR classifier", metrics["ear_ml_status"])

    if metrics.get("alert"):
        st.error(metrics.get("alert_message") or "Drowsiness indicator detected.")
    else:
        st.success("No active drowsiness alert.")


render_live_metrics()

st.divider()
st.caption(
    "The video overlay and metrics are generated from the incoming webcam "
    "stream. A STUN server is configured by default; some networks may also "
    "require TURN credentials in Streamlit Secrets."
)
