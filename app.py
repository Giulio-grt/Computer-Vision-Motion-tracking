import time
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# streamlit-webrtc imports
try:
    import av  # type: ignore
    from streamlit_webrtc import (
        RTCConfiguration,
        WebRtcMode,
        VideoProcessorBase,
        webrtc_streamer,
    )
except Exception as e:  # pragma: no cover - helpful message when deps missing
    av = None
    RTCConfiguration = None
    WebRtcMode = None
    VideoProcessorBase = object  # placeholder so type checker is happy
    webrtc_streamer = None


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


def put_fps_bgr(img: np.ndarray, fps: float) -> None:
    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


@dataclass
class ProcessorConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    view: str = "Stickman + Camera"  # Combined | Stickman + Camera | Stickman only | Camera only
    layout: str = "Side-by-side"  # or "Stacked"
    show_fps: bool = True
    mirror_selfie: bool = True
    stick_bg: str = "Black"  # or "White"
    scale_combined: bool = True  # shrink each panel to keep total size


class PoseProcessor(VideoProcessorBase):
    def __init__(self, config: ProcessorConfig):
        self.cfg = config
        self.pose = mp_pose.Pose(
            model_complexity=self.cfg.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self._prev_t = time.time()
        self._fps = 0.0

    def update_config(self, config: ProcessorConfig):
        # Recreate pose if complexity or thresholds changed significantly
        recreate = (
            config.model_complexity != self.cfg.model_complexity
            or abs(config.min_detection_confidence - self.cfg.min_detection_confidence) > 1e-6
            or abs(config.min_tracking_confidence - self.cfg.min_tracking_confidence) > 1e-6
        )
        self.cfg = config
        if recreate:
            if self.pose is not None:
                self.pose.close()
            self.pose = mp_pose.Pose(
                model_complexity=self.cfg.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=self.cfg.min_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
            )

    def _compute_fps(self):
        now = time.time()
        dt = now - self._prev_t
        if dt > 0:
            # Light smoothing
            curr = 1.0 / dt
            self._fps = 0.9 * self._fps + 0.1 * curr if self._fps > 0 else curr
        self._prev_t = now

    def recv(self, frame):  # frame: av.VideoFrame
        img_bgr = frame.to_ndarray(format="bgr24")

        if self.cfg.mirror_selfie:
            img_bgr = cv2.flip(img_bgr, 1)

        h, w = img_bgr.shape[:2]

        # Camera-only mode: skip pose inference entirely for lowest latency
        if self.cfg.view == "Camera only":
            if self.cfg.show_fps:
                self._compute_fps()
                put_fps_bgr(img_bgr, self._fps)
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        # Prepare stickman canvas
        if self.cfg.stick_bg == "White":
            stick = np.full((h, w, 3), 255, dtype=np.uint8)
        else:
            stick = np.zeros((h, w, 3), dtype=np.uint8)

        # Run pose inference once
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)
        rgb.flags.writeable = True

        overlay = None
        if self.cfg.view in ("Combined",):
            overlay = img_bgr.copy()

        if res.pose_landmarks:
            # Draw on stickman
            mp_draw.draw_landmarks(
                stick,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
            )
            # Draw on overlay if needed
            if overlay is not None:
                mp_draw.draw_landmarks(
                    overlay,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
                )

        # FPS overlay
        if self.cfg.show_fps:
            self._compute_fps()
            put_fps_bgr(stick, self._fps)
            if overlay is not None:
                put_fps_bgr(overlay, self._fps)

        # Output selection
        if self.cfg.view == "Stickman only":
            return av.VideoFrame.from_ndarray(stick, format="bgr24")

        # Two-panel views
        if self.cfg.view == "Stickman + Camera":
            left_panel = img_bgr
            right_panel = stick
        else:  # Combined (camera with landmarks + stickman)
            if overlay is None:
                overlay = img_bgr
            left_panel = overlay
            right_panel = stick

        # Optionally scale each panel to keep total size
        if self.cfg.scale_combined:
            new_w, new_h = int(w * 0.5), int(h * 0.5)
            left_s = cv2.resize(left_panel, (new_w, new_h), interpolation=cv2.INTER_AREA)
            right_s = cv2.resize(right_panel, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            left_s, right_s = left_panel, right_panel

        if self.cfg.layout == "Stacked":
            combined = np.vstack([left_s, right_s])
        else:
            combined = np.hstack([left_s, right_s])

        return av.VideoFrame.from_ndarray(combined, format="bgr24")


def main():
    st.set_page_config(page_title="MediaPipe Pose (Streamlit)", layout="wide")
    st.title("MediaPipe Pose — Live Camera with Stickman View")
    st.caption(
        "Left: camera with landmarks. Right: stickman only (or stacked)."
    )

    with st.sidebar:
        st.header("Settings")
        view = st.radio("View", ["Stickman + Camera", "Combined", "Stickman only", "Camera only"], index=0)
        layout = st.radio("Layout", ["Side-by-side", "Stacked"], index=0)
        mirror = st.checkbox("Mirror (selfie)", value=True)
        show_fps = st.checkbox("Show FPS", value=True)
        stick_bg = st.selectbox("Stickman Background", ["Black", "White"], index=0)
        async_proc = st.checkbox(
            "Async processing (threaded)", value=False,
            help="Can increase throughput but may add queue latency",
        )

        st.subheader("Model")
        complexity = st.selectbox("Model complexity", [0, 1, 2], index=1, help="Higher is more accurate but slower")
        det_conf = st.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.05)
        track_conf = st.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.05)

        st.subheader("Camera")
        res = st.selectbox("Resolution", [
            (640, 480), (1280, 720), (1920, 1080)
        ], index=1, format_func=lambda s: f"{s[0]}x{s[1]}")
        scale_combined = st.checkbox(
            "Keep combined size (half-scale per panel)",
            value=True,
            help="Applies to two-panel views; reduces pixels to lower latency",
        )

    # Prepare RTC configuration and constraints
    if RTCConfiguration is None or webrtc_streamer is None:
        st.error(
            "Missing dependencies. Please install: pip install streamlit streamlit-webrtc av mediapipe opencv-python"
        )
        return

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    })

    media_constraints = {
        "video": {
            "width": {"ideal": int(res[0])},
            "height": {"ideal": int(res[1])},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    }

    base_cfg = ProcessorConfig(
        view=str(view),
        model_complexity=int(complexity),
        min_detection_confidence=float(det_conf),
        min_tracking_confidence=float(track_conf),
        layout=str(layout),
        show_fps=bool(show_fps),
        mirror_selfie=bool(mirror),
        stick_bg=str(stick_bg),
        scale_combined=bool(scale_combined),
    )

    def factory() -> PoseProcessor:
        return PoseProcessor(config=base_cfg)

    ctx = webrtc_streamer(
        key="pose-demo",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints=media_constraints,
        video_processor_factory=factory,
        async_processing=bool(async_proc),
    )

    # If UI changes, push updates into the existing processor instance
    if ctx.video_processor:
        ctx.video_processor.update_config(
            ProcessorConfig(
                view=str(view),
                model_complexity=int(complexity),
                min_detection_confidence=float(det_conf),
                min_tracking_confidence=float(track_conf),
                layout=str(layout),
                show_fps=bool(show_fps),
                mirror_selfie=bool(mirror),
                stick_bg=str(stick_bg),
                scale_combined=bool(scale_combined),
            )
        )

    st.info(
        "Click Start, allow camera access. Use the sidebar to tweak settings."
    )


if __name__ == "__main__":
    main()
