"""Streamlit dashboard for the fatigue detection system."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import winsound
except ImportError:  # pragma: no cover - Windows-only dependency
    winsound = None

from core.fatigue_scorer import FatigueLevel
from main_detector import RealtimeDetector
from system.screen_control import ScreenController
from system.session_logger import SessionLogger

PROJECT_ROOT = Path(__file__).parent
FRAME_DELAY_MS = 30
HISTORY_SIZE = 300
CHART_REFRESH_INTERVAL = 6

LEVEL_META = {
    FatigueLevel.ALERT: {
        "label": "Alert",
        "tone": "safe",
        "color": "#2e8b57",
        "message": "User looks stable and attentive.",
    },
    FatigueLevel.MILD: {
        "label": "Mild Fatigue",
        "tone": "warn",
        "color": "#d97706",
        "message": "Early fatigue signs detected. Slow down and stay focused.",
    },
    FatigueLevel.MODERATE: {
        "label": "Moderate Fatigue",
        "tone": "warn",
        "color": "#ea580c",
        "message": "Fatigue is building. A break is strongly recommended.",
    },
    FatigueLevel.SEVERE: {
        "label": "Severe Fatigue",
        "tone": "danger",
        "color": "#dc2626",
        "message": "Immediate rest is recommended before continuing.",
    },
}


st.set_page_config(
    page_title="Fatigue Detection Dashboard",
    page_icon=":eye:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.2rem;
    }
    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #38bdf8 100%);
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        opacity: 0.92;
        font-size: 1rem;
    }
    .status-card {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: white;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
    }
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .pill-safe {
        background: rgba(46, 139, 87, 0.14);
        color: #166534;
    }
    .pill-warn {
        background: rgba(217, 119, 6, 0.14);
        color: #9a3412;
    }
    .pill-danger {
        background: rgba(220, 38, 38, 0.14);
        color: #991b1b;
    }
    .info-panel {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
        border: 1px solid #dbeafe;
        margin-bottom: 1rem;
    }
    .metric-strip {
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _new_metric_history() -> dict[str, deque]:
    return {
        "score": deque(maxlen=HISTORY_SIZE),
        "ear": deque(maxlen=HISTORY_SIZE),
        "perclos": deque(maxlen=HISTORY_SIZE),
        "blink_rate": deque(maxlen=HISTORY_SIZE),
        "cnn_fatigue": deque(maxlen=HISTORY_SIZE),
        "cnn_eye": deque(maxlen=HISTORY_SIZE),
        "fps": deque(maxlen=HISTORY_SIZE),
        "elapsed_seconds": deque(maxlen=HISTORY_SIZE),
    }


def init_session_state() -> None:
    defaults = {
        "detector": None,
        "camera": None,
        "running": False,
        "fatigue_detected": False,
        "alert_count": 0,
        "frame_index": 0,
        "session_start": None,
        "screen_controller": ScreenController(),
        "session_logger": None,
        "metric_history": _new_metric_history(),
        "last_result": None,
        "last_frame_rgb": None,
        "brightness_lowered": False,
        "alert_started_at": None,
        "last_error": None,
        "last_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_level_meta(level: FatigueLevel | None) -> dict[str, str]:
    if level is None:
        return {
            "label": "Waiting",
            "tone": "warn",
            "color": "#475569",
            "message": "Start a session to begin live monitoring.",
        }
    return LEVEL_META.get(level, LEVEL_META[FatigueLevel.ALERT])


def reset_metric_history() -> None:
    st.session_state.metric_history = _new_metric_history()


def release_camera() -> None:
    camera = st.session_state.camera
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    st.session_state.camera = None


def restore_brightness() -> None:
    try:
        st.session_state.screen_controller.set_brightness(100)
    except Exception:
        pass
    st.session_state.brightness_lowered = False


def end_logger_session() -> None:
    logger = st.session_state.session_logger
    if logger is None:
        return
    try:
        st.session_state.last_summary = logger.end_session()
    except Exception as exc:
        st.session_state.last_error = f"Failed to save session summary: {exc}"
    finally:
        st.session_state.session_logger = None


def stop_detection(error_message: str | None = None) -> None:
    st.session_state.running = False
    st.session_state.fatigue_detected = False
    st.session_state.alert_started_at = None
    release_camera()
    if st.session_state.brightness_lowered:
        restore_brightness()
    end_logger_session()
    if error_message:
        st.session_state.last_error = error_message


def play_alert_sound() -> None:
    if winsound is None:
        return
    try:
        winsound.Beep(900, 180)
        winsound.Beep(1150, 220)
    except Exception:
        pass


def initialize_detector() -> bool:
    if st.session_state.detector is not None:
        return True

    with st.spinner("Loading detection models..."):
        try:
            detector = RealtimeDetector()
            detector.setup_gpu()
            if not detector.load_models():
                st.session_state.last_error = "Failed to load fatigue models."
                return False
            st.session_state.detector = detector
            return True
        except Exception as exc:
            st.session_state.last_error = f"Failed to initialize detector: {exc}"
            return False


def apply_thresholds(alert_threshold: float) -> None:
    detector = st.session_state.detector
    if detector is None:
        return
    moderate = min(max(alert_threshold + 0.22, 0.55), 0.82)
    severe = min(max(moderate + 0.18, 0.75), 0.95)
    detector.fatigue_scorer.set_thresholds(
        mild=alert_threshold,
        moderate=moderate,
        severe=severe,
    )


def start_detection(camera_index: int, alert_threshold: float) -> bool:
    st.session_state.last_error = None
    st.session_state.last_summary = None

    if not initialize_detector():
        return False

    apply_thresholds(alert_threshold)
    release_camera()

    camera = st.session_state.detector.open_webcam([camera_index])
    if camera is None or not camera.isOpened():
        st.session_state.last_error = f"Could not open webcam at index {camera_index}."
        return False

    st.session_state.camera = camera
    st.session_state.running = True
    st.session_state.fatigue_detected = False
    st.session_state.brightness_lowered = False
    st.session_state.alert_count = 0
    st.session_state.frame_index = 0
    st.session_state.session_start = time.time()
    st.session_state.detector.frame_count = 0
    st.session_state.detector.fps_counter = 0
    st.session_state.detector.current_fps = 0
    st.session_state.detector.fps_timer = time.time()
    st.session_state.detector.session_start = st.session_state.session_start
    st.session_state.alert_started_at = None
    st.session_state.last_result = None
    st.session_state.last_frame_rgb = None
    st.session_state.session_logger = SessionLogger()
    reset_metric_history()
    return True


def append_metric(result: dict, fps: float) -> None:
    history = st.session_state.metric_history
    face_metrics = result["face_metrics"]
    elapsed = 0.0
    if st.session_state.session_start is not None:
        elapsed = time.time() - st.session_state.session_start

    history["score"].append(float(result.get("fatigue_score", 0.0)))
    history["ear"].append(float(face_metrics.get("ear", 0.0)))
    history["perclos"].append(float(face_metrics.get("perclos", 0.0)))
    history["blink_rate"].append(float(face_metrics.get("blink_rate", 0.0)))
    history["cnn_fatigue"].append(
        float(result["cnn_fatigue"]) if result.get("cnn_fatigue") is not None else np.nan
    )
    history["cnn_eye"].append(
        float(result["cnn_eye"]) if result.get("cnn_eye") is not None else np.nan
    )
    history["fps"].append(float(fps))
    history["elapsed_seconds"].append(float(elapsed))


def update_alert_state(level: FatigueLevel, target_brightness: int) -> None:
    fatigued = level != FatigueLevel.ALERT
    logger = st.session_state.session_logger

    if fatigued and not st.session_state.fatigue_detected:
        st.session_state.fatigue_detected = True
        st.session_state.alert_count += 1
        st.session_state.alert_started_at = time.time()
        play_alert_sound()
        try:
            st.session_state.screen_controller.set_brightness(target_brightness)
            st.session_state.brightness_lowered = target_brightness < 100
        except Exception:
            st.session_state.brightness_lowered = False
        if logger is not None:
            logger.log_event("fatigue_alert", f"Fatigue escalated to {level.name}")

    elif not fatigued and st.session_state.fatigue_detected:
        st.session_state.fatigue_detected = False
        st.session_state.alert_started_at = None
        if st.session_state.brightness_lowered:
            restore_brightness()
        if logger is not None:
            logger.log_event("recovered", "User returned to alert state")

    elif fatigued and st.session_state.brightness_lowered and target_brightness < 100:
        try:
            st.session_state.screen_controller.set_brightness(target_brightness)
        except Exception:
            pass


def process_one_frame(target_brightness: int, alert_threshold: float) -> bool:
    detector = st.session_state.detector
    camera = st.session_state.camera
    if detector is None or camera is None:
        stop_detection("Camera session is not available.")
        return False

    apply_thresholds(alert_threshold)

    ok, frame = camera.read()
    if not ok:
        stop_detection("Failed to read a frame from the webcam.")
        return False

    result = detector.process_frame(frame)
    detector.frame_count += 1
    detector.update_fps()
    annotated = detector.draw_metrics_overlay(frame.copy(), result)
    st.session_state.last_frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.session_state.last_result = result

    append_metric(result, detector.current_fps)

    logger = st.session_state.session_logger
    if logger is not None:
        logger.log_frame(
            st.session_state.frame_index,
            result["fatigue_score"],
            result["level"],
            result["face_metrics"],
            result.get("cnn_fatigue"),
            result.get("cnn_eye"),
        )
    st.session_state.frame_index += 1

    update_alert_state(result["level"], target_brightness)
    return True


def render_status_panel(rest_duration: int) -> None:
    result = st.session_state.last_result
    meta = get_level_meta(result["level"] if result else None)
    pill_class = {
        "safe": "pill-safe",
        "warn": "pill-warn",
        "danger": "pill-danger",
    }[meta["tone"]]

    message = meta["message"]
    if st.session_state.fatigue_detected and st.session_state.alert_started_at is not None:
        elapsed = int(time.time() - st.session_state.alert_started_at)
        remaining = max(rest_duration - elapsed, 0)
        message = f"{message} Suggested break window remaining: {remaining}s."

    st.markdown(
        f"""
        <div class="status-card">
            <span class="status-pill {pill_class}">{meta["label"]}</span>
            <h3 style="margin:0; color:{meta["color"]};">Live Safety Status</h3>
            <p style="margin:0.6rem 0 0 0; color:#334155;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)


def render_live_metrics() -> None:
    result = st.session_state.last_result
    if not result:
        st.info("Start detection to see live metrics.")
        return

    face_metrics = result["face_metrics"]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Fatigue Score", f"{result['fatigue_score']:.3f}")
        st.metric("EAR", f"{face_metrics.get('ear', 0.0):.3f}")
        st.metric("PERCLOS", f"{face_metrics.get('perclos', 0.0):.1f}%")
    with c2:
        st.metric("Blink Rate", f"{face_metrics.get('blink_rate', 0.0):.0f}/min")
        st.metric(
            "Face CNN",
            "N/A" if result.get("cnn_fatigue") is None else f"{result['cnn_fatigue']:.3f}",
        )
        st.metric(
            "Eye CNN (Open)",
            "N/A" if result.get("cnn_eye") is None else f"{result['cnn_eye']:.3f}",
        )


def render_charts() -> None:
    history = st.session_state.metric_history
    if not history["score"]:
        st.info("Run a session to populate the trend charts.")
        return

    if st.session_state.running and st.session_state.frame_index % CHART_REFRESH_INTERVAL != 0:
        st.info("Trend charts refresh every few frames during live detection to keep the stream responsive.")
        return

    df = pd.DataFrame(
        {
            "Seconds": list(history["elapsed_seconds"]),
            "Fatigue Score": list(history["score"]),
            "EAR": list(history["ear"]),
            "PERCLOS": list(history["perclos"]),
            "Blink Rate": list(history["blink_rate"]),
            "Face CNN": list(history["cnn_fatigue"]),
            "Eye CNN (Open)": list(history["cnn_eye"]),
            "FPS": list(history["fps"]),
        }
    )
    df = df.drop_duplicates(subset=["Seconds"], keep="last")
    df = df.set_index("Seconds")

    st.subheader("Fatigue Trend")
    st.line_chart(df[["Fatigue Score", "PERCLOS"]], height=320)

    stats = st.columns(4)
    stats[0].metric("Average Score", f"{np.nanmean(df['Fatigue Score']):.3f}")
    stats[1].metric("Peak Score", f"{np.nanmax(df['Fatigue Score']):.3f}")
    stats[2].metric("Average EAR", f"{np.nanmean(df['EAR']):.3f}")
    stats[3].metric("Average FPS", f"{np.nanmean(df['FPS']):.1f}")

    st.subheader("Supporting Signals")
    st.line_chart(df[["EAR", "Blink Rate", "Eye CNN (Open)"]], height=280)


def render_summary() -> None:
    summary = st.session_state.last_summary
    if not summary:
        st.info("A session summary will appear here after you stop detection.")
        return

    session_info = summary["session_info"]
    fatigue_stats = summary["fatigue_statistics"]
    metrics = summary["metrics"]

    cols = st.columns(3)
    cols[0].metric("Duration", f"{session_info['duration_minutes']} min")
    cols[1].metric("Frames Logged", f"{session_info['total_frames']}")
    cols[2].metric("Peak Score", f"{fatigue_stats['peak_fatigue_score']}")

    st.markdown(
        f"""
        <div class="info-panel">
            <strong>Session summary saved.</strong><br>
            CSV: {session_info['csv_file']}<br>
            Average score: {fatigue_stats['average_fatigue_score']}<br>
            Average EAR: {metrics['average_ear']}<br>
            Average blink rate: {metrics['average_blink_rate']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    events = summary.get("notable_events", [])
    if events:
        st.subheader("Recent Events")
        st.dataframe(pd.DataFrame(events), use_container_width=True)


def render_info() -> None:
    st.markdown(
        """
        ### How this dashboard works

        - The webcam is processed one frame at a time so the Streamlit UI stays responsive.
        - The detector combines face CNN output, eye-state CNN output, EAR, blink rate, and PERCLOS.
        - Fatigue alerts dim the screen once and then restore brightness after recovery.
        - Each session writes CSV logs plus a JSON summary into `logs/`.

        ### Practical controls

        - `Alert threshold` changes when the detector leaves the safe state.
        - `Target brightness` controls how much the screen dims during an alert.
        - `Camera index` helps if your laptop webcam is not on index `0`.
        """
    )


def main() -> None:
    init_session_state()

    st.markdown(
        """
        <div class="hero">
            <h1>Fatigue Detection Dashboard</h1>
            <p>Live fatigue scoring, clearer status feedback, responsive controls, and session logging.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
        alert_threshold = st.slider("Alert Threshold", 0.25, 0.70, 0.38, 0.01)
        target_brightness = st.slider("Target Brightness", 10, 100, 30, 5)
        rest_duration = st.slider("Rest Reminder (seconds)", 10, 90, 30, 5)
        st.caption("Live mode is tuned for lower latency: 640x480 capture, faster reruns, and lighter chart refresh.")

        st.markdown('<div class="metric-strip">', unsafe_allow_html=True)
        if st.session_state.session_start is not None:
            elapsed = int(time.time() - st.session_state.session_start) if st.session_state.running else 0
            st.metric("Session Time", f"{elapsed}s")
        st.metric("Alerts", st.session_state.alert_count)
        st.metric("Brightness Dimmed", "Yes" if st.session_state.brightness_lowered else "No")
        st.markdown("</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Live Monitor", "Trends", "Session Summary", "Info"])

    with tabs[0]:
        controls_col, feed_col = st.columns([1, 2.2])

        with controls_col:
            render_status_panel(rest_duration)

            if st.button(
                "Start Detection",
                type="primary",
                disabled=st.session_state.running,
                use_container_width=True,
            ):
                if start_detection(int(camera_index), float(alert_threshold)):
                    st.rerun()

            if st.button(
                "Stop Detection",
                disabled=not st.session_state.running,
                use_container_width=True,
            ):
                stop_detection()
                st.rerun()

            if st.button("Restore Brightness", use_container_width=True):
                restore_brightness()

            if st.button("Clear History", use_container_width=True):
                reset_metric_history()
                st.session_state.last_result = None
                st.session_state.last_frame_rgb = None

            render_live_metrics()

        with feed_col:
            st.subheader("Webcam Feed")
            if st.session_state.last_frame_rgb is not None:
                st.image(
                    st.session_state.last_frame_rgb,
                    channels="RGB",
                    use_column_width=True,
                )
            else:
                st.info("No webcam frame yet. Start detection to begin streaming.")

    with tabs[1]:
        render_charts()

    with tabs[2]:
        render_summary()

    with tabs[3]:
        render_info()

    if st.session_state.running:
        processed = process_one_frame(int(target_brightness), float(alert_threshold))
        if processed and st.session_state.running:
            time.sleep(FRAME_DELAY_MS / 1000)
            st.rerun()


if __name__ == "__main__":
    main()
