import cv2, time, numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def put_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
 
cap = cv2.VideoCapture(0)  # try 1 if you have multiple cameras
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

# --- set higher resolution (depends on your camera capability) ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Full HD width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height

# --- make window resizable and large ---
cv2.namedWindow("MediaPipe Pose (q to quit)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MediaPipe Pose (q to quit)", 1920, 1080)

prev = time.time()
with mp_pose.Pose(model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok: break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        res = pose.process(image)

        frame.flags.writeable = True
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_style.get_default_pose_landmarks_style()
            )

        now = time.time()
        fps = 1.0/(now - prev) if now > prev else 0.0
        prev = now
        put_fps(frame, fps)
        cv2.imshow("MediaPipe Pose (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

