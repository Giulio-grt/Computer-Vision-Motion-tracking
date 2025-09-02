import cv2, csv, math
import numpy as np
import mediapipe as mp

INPUT = "assets/input.mp4"
OUT_VIDEO = "annotated.mp4"
OUT_CSV = "metrics_3d.csv"

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ---------- 3D vector helpers ----------
def vsub(a, b):  # a - b
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def vdot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vnorm(a):
    return math.sqrt(vdot(a, a))

def vunit(a):
    n = vnorm(a)
    if n == 0: return (0.0, 0.0, 0.0)
    return (a[0]/n, a[1]/n, a[2]/n)

def vcross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

# angle at vertex B between BA and BC (3D)
def angle3d(a, b, c):
    BA = vsub(a, b)
    BC = vsub(c, b)
    nBA, nBC = vnorm(BA), vnorm(BC)
    if nBA == 0 or nBC == 0:
        return None
    cosv = max(-1.0, min(1.0, vdot(BA, BC) / (nBA * nBC)))
    return math.degrees(math.acos(cosv))

# ---------- 2D helper for drawing overlay ----------
def get_xy(lms, idx, w, h):
    lm = lms[idx]
    return (lm.x * w, lm.y * h)

# ---------- Get world (x,y,z) in meters ----------
def get_xyz_world(wlms, idx):
    lm = wlms[idx]
    return (lm.x, lm.y, lm.z)

# ---------- Build a torso/body frame and get yaw/pitch/roll ----------
"""
We define a simple body frame:

- shoulders_mid = average of L/R shoulders
- hips_mid      = average of L/R hips
- up axis  ŷ_b  = unit(shoulders_mid - hips_mid)    (roughly vertical along torso)
- right axis x̂_b= unit(R_shoulder - L_shoulder)    (points to athlete's right)
- forward  ẑ_b  = x̂_b × ŷ_b                       (right-hand rule)

To express orientation as yaw/pitch/roll relative to world frame:
We consider world axes as MediaPipe's world (approximately: x right, y up, z forward/back).
We compute a rotation matrix R whose columns are [x̂_b, ŷ_b, ẑ_b] in world coords,
then extract Euler angles in Z (yaw), X (pitch), Y (roll) order (ZYX convention).
"""
def torso_orientation(world):
    # indices
    LS, RS = 11, 12
    LH, RH = 23, 24
    pLS = get_xyz_world(world, LS)
    pRS = get_xyz_world(world, RS)
    pLH = get_xyz_world(world, LH)
    pRH = get_xyz_world(world, RH)
    shoulders_mid = ((pLS[0]+pRS[0])/2.0, (pLS[1]+pRS[1])/2.0, (pLS[2]+pRS[2])/2.0)
    hips_mid      = ((pLH[0]+pRH[0])/2.0, (pLH[1]+pRH[1])/2.0, (pLH[2]+pRH[2])/2.0)

    yb = vunit( vsub(shoulders_mid, hips_mid) )   # up (torso axis)
    xb = vunit( vsub(pRS, pLS) )                  # right (shoulder line)
    zb = vunit( vcross(xb, yb) )                  # forward (right-hand)
    # re-orthogonalize x using y×z in case of slight drift
    xb = vunit( vcross(yb, zb) )

    # Rotation matrix R (world <- body), columns are body axes in world frame
    R = np.array([[xb[0], yb[0], zb[0]],
                  [xb[1], yb[1], zb[1]],
                  [xb[2], yb[2], zb[2]]], dtype=float)

    # Extract ZYX Euler: yaw(Z), pitch(X), roll(Y)
    # yaw   = atan2(R21, R11)  -> using standard formulas for ZYX
    # pitch = asin(-R31)
    # roll  = atan2(R32, R33)
    # (Here rows/cols are 0-based)
    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    yaw   = math.degrees(math.atan2(r21, r11))
    pitch = math.degrees(math.asin(-r31))
    roll  = math.degrees(math.atan2(r32, r33))

    # root (pelvis) position = hips_mid
    root = hips_mid
    return yaw, pitch, roll, root

# ---------- landmark lists ----------
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

# ---------- Video IO ----------
cap = cv2.VideoCapture(INPUT)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {INPUT}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (W, H))

with open(OUT_CSV, "w", newline="") as fcsv, \
     mp_pose.Pose(model_complexity=2,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    writer = csv.writer(fcsv)
    # Header: time + 3D angles + torso yaw/pitch/roll + pelvis/root + 3D joints
    header = [
        "time_s",
        "knee_left_deg_3d", "knee_right_deg_3d",
        "elbow_left_deg_3d", "elbow_right_deg_3d",
        "torso_yaw_deg", "torso_pitch_deg", "torso_roll_deg",
        "root_x_m", "root_y_m", "root_z_m"
    ]
    header += [f"{n}_x" for n in body_landmark_names]
    header += [f"{n}_y" for n in body_landmark_names]
    header += [f"{n}_z" for n in body_landmark_names]
    writer.writerow(header)

    t = 0.0
    dt = 1.0 / fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)

        if res.pose_landmarks:
            # Draw 2D overlay for visualization
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )

        # If we have world landmarks, compute 3D metrics
        if res.pose_world_landmarks:
            wlms = res.pose_world_landmarks.landmark

            # ---- 3D joint angles (knees, elbows) ----
            # knees: angle at knee with hip-knee-ankle
            L_HIP, R_HIP = 23, 24
            L_KNEE, R_KNEE = 25, 26
            L_ANK, R_ANK = 27, 28
            L_SH, R_SH = 11, 12
            L_ELB, R_ELB = 13, 14
            L_WRI, R_WRI = 15, 16

            pLH = get_xyz_world(wlms, L_HIP)
            pRH = get_xyz_world(wlms, R_HIP)
            pLK = get_xyz_world(wlms, L_KNEE)
            pRK = get_xyz_world(wlms, R_KNEE)
            pLA = get_xyz_world(wlms, L_ANK)
            pRA = get_xyz_world(wlms, R_ANK)

            pLS = get_xyz_world(wlms, L_SH)
            pRS = get_xyz_world(wlms, R_SH)
            pLE = get_xyz_world(wlms, L_ELB)
            pRE = get_xyz_world(wlms, R_ELB)
            pLW = get_xyz_world(wlms, L_WRI)
            pRW = get_xyz_world(wlms, R_WRI)

            knee_l_3d = angle3d(pLH, pLK, pLA)  # at left knee
            knee_r_3d = angle3d(pRH, pRK, pRA)  # at right knee
            elbow_l_3d = angle3d(pLS, pLE, pLW) # at left elbow
            elbow_r_3d = angle3d(pRS, pRE, pRW) # at right elbow

            # ---- Torso orientation + root position ----
            yaw, pitch, roll, root = torso_orientation(wlms)

            # ---- Collect all world joints for CSV ----
            Xs, Ys, Zs = [], [], []
            for i in body_indices:
                x, y, z = get_xyz_world(wlms, i)
                Xs.append(round(x, 6))
                Ys.append(round(y, 6))
                Zs.append(round(z, 6))

            # ---- Overlay some numbers (optional) ----
            if res.pose_landmarks:
                def put(txt, y):
                    cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                put(f"t={t:.2f}s", 30)
                put(f"Knee3D L/R: {None if knee_l_3d is None else round(knee_l_3d,1)} / "
                    f"{None if knee_r_3d is None else round(knee_r_3d,1)}", 60)
                put(f"Elbow3D L/R: {None if elbow_l_3d is None else round(elbow_l_3d,1)} / "
                    f"{None if elbow_r_3d is None else round(elbow_r_3d,1)}", 90)
                put(f"Torso Y/P/R: {round(yaw,1)}, {round(pitch,1)}, {round(roll,1)}", 120)

            # ---- Write CSV row ----
            row = [
                f"{t:.3f}",
                "" if knee_l_3d is None else round(knee_l_3d, 3),
                "" if knee_r_3d is None else round(knee_r_3d, 3),
                "" if elbow_l_3d is None else round(elbow_l_3d, 3),
                "" if elbow_r_3d is None else round(elbow_r_3d, 3),
                round(yaw,   3), round(pitch, 3), round(roll,  3),
                round(root[0], 6), round(root[1], 6), round(root[2], 6)
            ]
            writer.writerow(row + Xs + Ys + Zs)

        # Write annotated frame and advance time
        out.write(frame)
        t += dt

cap.release()
out.release()
print(f"Saved: {OUT_VIDEO}, {OUT_CSV}")
