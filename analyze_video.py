import cv2, csv, math
import numpy as np
import mediapipe as mp

INPUT = "input.mp4"
OUT_VIDEO = "annotated.mp4"
OUT_CSV = "metrics.csv"

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Utility: angle between three points (in degrees) with B at the vertex
def angle(a, b, c):
    # a,b,c are (x,y)
    ab = (a[0]-b[0], a[1]-b[1])
    cb = (c[0]-b[0], c[1]-b[1])
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    nab = math.hypot(*ab); ncb = math.hypot(*cb)
    if nab*ncb == 0: return None
    cosv = max(-1.0, min(1.0, dot/(nab*ncb)))
    return math.degrees(math.acos(cosv))

# Landmark helper
def get_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

cap = cv2.VideoCapture(INPUT)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {INPUT}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

with open(OUT_CSV, "w", newline="") as fcsv, \
     mp_pose.Pose(model_complexity=2, enable_segmentation=False,
                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    writer = csv.writer(fcsv)

    # ---- ONLY BODY LANDMARKS (indices 11..32) ----
    body_landmark_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    body_indices = list(range(11, 33))  # 11..32 inclusive

    # CSV header: time + 4 angles + body (x then y)
    header = (["time_s","knee_left_deg","knee_right_deg","elbow_left_deg","elbow_right_deg"] +
              [f"{n}_x" for n in body_landmark_names] +
              [f"{n}_y" for n in body_landmark_names])
    writer.writerow(header)
    # -----------------------------------------------

    t = 0.0
    dt = 1.0 / fps

    while True:
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)

        if res.pose_landmarks:
            # draw
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )

            lms = res.pose_landmarks.landmark
            # collect all pixel coords
            pts = {}
            for idx in range(33):
                pts[idx] = get_xy(lms, idx, w, h)

            # Angles
            knee_l = angle(pts[23], pts[25], pts[27]) if 23 in pts and 25 in pts and 27 in pts else None
            knee_r = angle(pts[24], pts[26], pts[28]) if 24 in pts and 26 in pts and 28 in pts else None
            elbow_l = angle(pts[11], pts[13], pts[15]) if 11 in pts and 13 in pts and 15 in pts else None
            elbow_r = angle(pts[12], pts[14], pts[16]) if 12 in pts and 14 in pts and 16 in pts else None

            # Overlay numbers
            def put(txt, y): cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            put(f"t={t:.2f}s", 30)
            put(f"Knee L/R: {knee_l and round(knee_l,1)} / {knee_r and round(knee_r,1)}", 60)
            put(f"Elbow L/R: {elbow_l and round(elbow_l,1)} / {elbow_r and round(elbow_r,1)}", 90)

            # ---- WRITE ROW: time, 4 angles, body coords (x then y) ----
            row = [
                f"{t:.3f}",
                f"{'' if knee_l is None else round(knee_l,3)}",
                f"{'' if knee_r is None else round(knee_r,3)}",
                f"{'' if elbow_l is None else round(elbow_l,3)}",
                f"{'' if elbow_r is None else round(elbow_r,3)}"
            ]
            xs = [f"{'' if pts.get(i) is None else round(pts[i][0],3)}" for i in body_indices]
            ys = [f"{'' if pts.get(i) is None else round(pts[i][1],3)}" for i in body_indices]
            writer.writerow(row + xs + ys)
            # -----------------------------------------------------------

        out.write(frame)
        t += dt

cap.release()
out.release()
print(f"Saved: {OUT_VIDEO}, {OUT_CSV}")
