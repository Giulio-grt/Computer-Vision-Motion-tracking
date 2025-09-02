# play_skeleton_2d.py
import sys, os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

def main(csv_path):
    if not os.path.exists(csv_path):
        raise SystemExit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    JOINTS_ALL = [
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
        "left_foot_index", "right_foot_index",
    ]

    def cols_exist(j): 
        return all(f"{j}_{ax}" in df.columns for ax in ("x","y"))
    JOINTS = [j for j in JOINTS_ALL if cols_exist(j)]
    if not JOINTS:
        raise SystemExit("No <joint>_{x,y} columns found in the CSV.")

    # Bone list (pairs)
    BONES = [
        ("left_shoulder","right_shoulder"),
        ("left_hip","right_hip"),

        ("left_shoulder","left_elbow"),
        ("left_elbow","left_wrist"),

        ("right_shoulder","right_elbow"),
        ("right_elbow","right_wrist"),

        ("left_wrist","left_index"),
        ("left_wrist","left_pinky"),
        ("left_wrist","left_thumb"),
        ("right_wrist","right_index"),
        ("right_wrist","right_pinky"),
        ("right_wrist","right_thumb"),

        ("left_shoulder","left_hip"),
        ("right_shoulder","right_hip"),

        ("left_hip","left_knee"),
        ("left_knee","left_ankle"),
        ("left_ankle","left_heel"),
        ("left_heel","left_foot_index"),

        ("right_hip","right_knee"),
        ("right_knee","right_ankle"),
        ("right_ankle","right_heel"),
        ("right_heel","right_foot_index"),
    ]

    # === MediaPipe-style coloring ===
    LEFT_JOINTS = {
        "left_shoulder","left_elbow","left_wrist","left_pinky","left_index","left_thumb",
        "left_hip","left_knee","left_ankle","left_heel","left_foot_index"
    }
    RIGHT_JOINTS = {
        "right_shoulder","right_elbow","right_wrist","right_pinky","right_index","right_thumb",
        "right_hip","right_knee","right_ankle","right_heel","right_foot_index"
    }
    COLORS = {
        "left": "#90ee90",   # pastel green
        "right": "#ffcc80",  # pastel orange
        "center": "#aaaaaa", 
    }
    def bone_side(a, b):
        if a in LEFT_JOINTS and b in LEFT_JOINTS:
            return "left"
        if a in RIGHT_JOINTS and b in RIGHT_JOINTS:
            return "right"
        return "center"

    # Axis bounds fixed for all frames
    allX, allY = [], []
    for j in JOINTS:
        allX.append(pd.to_numeric(df[f"{j}_x"], errors="coerce").to_numpy())
        allY.append(pd.to_numeric(df[f"{j}_y"], errors="coerce").to_numpy())
    allX = np.hstack(allX); allY = np.hstack(allY)
    finite_mask = np.isfinite(allX) & np.isfinite(allY)
    if not finite_mask.any():
        raise SystemExit("No finite 2D points found in the CSV.")

    xmin, xmax = np.nanmin(allX[finite_mask]), np.nanmax(allX[finite_mask])
    ymin, ymax = np.nanmin(allY[finite_mask]), np.nanmax(allY[finite_mask])

    def pad(a, b, p=0.05):
        d = (b - a) or 1.0
        return a - p*d, b + p*d
    xr, yr = pad(xmin, xmax), pad(ymin, ymax)

    # Frame helpers
    def frame_segments(i):
        xs_left, ys_left = [], []
        xs_right, ys_right = [], []
        xs_center, ys_center = [], []
        row = df.iloc[i]
        for a, b in BONES:
            if not (cols_exist(a) and cols_exist(b)):
                continue
            ax, ay = row.get(f"{a}_x", np.nan), row.get(f"{a}_y", np.nan)
            bx, by = row.get(f"{b}_x", np.nan), row.get(f"{b}_y", np.nan)
            if np.all(np.isfinite([ax, ay, bx, by])):
                s = bone_side(a, b)
                if s == "left":
                    xs_left += [ax, bx, None]
                    ys_left += [ay, by, None]
                elif s == "right":
                    xs_right += [ax, bx, None]
                    ys_right += [ay, by, None]
                else:
                    xs_center += [ax, bx, None]
                    ys_center += [ay, by, None]
        return (xs_left, ys_left), (xs_right, ys_right), (xs_center, ys_center)

    def frame_points(i):
        px, py, txt = [], [], []
        row = df.iloc[i]
        for j in JOINTS:
            x, y = row.get(f"{j}_x", np.nan), row.get(f"{j}_y", np.nan)
            if np.all(np.isfinite([x, y])):
                px.append(x); py.append(y); txt.append(j)
        return px, py, txt

    # Timing
    t = pd.to_numeric(df.get("time_s", pd.Series(index=df.index, dtype=float)), errors="coerce").to_numpy()
    if not np.isfinite(t).any():
        t = np.arange(len(df), dtype=float)
    dts = np.diff(t[np.isfinite(t)])
    duration_ms = int(max(1, np.median(dts)*1000)) if dts.size else 33

    # Series for the time plot
    need_cols = ["right_wrist_x", "left_ankle_x"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"Column '{c}' not found in CSV. Available columns: {list(df.columns)}")

    rw_x = pd.to_numeric(df["right_wrist_x"], errors="coerce").to_numpy()
    la_x = pd.to_numeric(df["left_ankle_x"], errors="coerce").to_numpy()

    # --- Figure with subplots: skeleton + time-series
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.08,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("2D Pose Replay", "X vs Time")
    )

    # Initial skeleton frame
    (xsL0, ysL0), (xsR0, ysR0), (xsC0, ysC0) = frame_segments(0)
    px0, py0, txt0 = frame_points(0)

    bones_left = go.Scatter(x=xsL0, y=ysL0, mode="lines",
                            line=dict(width=3, color=COLORS["left"]),
                            showlegend=False, hoverinfo="skip")
    bones_right = go.Scatter(x=xsR0, y=ysR0, mode="lines",
                             line=dict(width=3, color=COLORS["right"]),
                             showlegend=False, hoverinfo="skip")
    bones_center = go.Scatter(x=xsC0, y=ysC0, mode="lines",
                              line=dict(width=3, color=COLORS["center"]),
                              showlegend=False, hoverinfo="skip")
    points_tr = go.Scatter(x=px0, y=py0, mode="markers",
                           marker=dict(size=6),
                           hovertext=txt0, hoverinfo="text",
                           showlegend=False)

    fig.add_trace(bones_left,   row=1, col=1)   # idx 0
    fig.add_trace(bones_right,  row=1, col=1)   # idx 1
    fig.add_trace(bones_center, row=1, col=1)   # idx 2
    fig.add_trace(points_tr,    row=1, col=1)   # idx 3

    # Time-series lines with same colors
    rw_line = go.Scatter(x=t, y=rw_x, mode="lines", name="right_wrist_x", line=dict(color=COLORS["right"]))
    la_line = go.Scatter(x=t, y=la_x, mode="lines", name="left_ankle_x", line=dict(color=COLORS["left"]))
    fig.add_trace(rw_line, row=1, col=2)        # idx 4
    fig.add_trace(la_line, row=1, col=2)        # idx 5

    # Moving dots with same colors
    t0 = t[0] if len(t) else 0.0
    rw_dot = go.Scatter(x=[t0], y=[rw_x[0] if len(rw_x) else np.nan],
                        mode="markers", marker=dict(size=10, color=COLORS["right"]),
                        showlegend=False)
    la_dot = go.Scatter(x=[t0], y=[la_x[0] if len(la_x) else np.nan],
                        mode="markers", marker=dict(size=10, color=COLORS["left"]),
                        showlegend=False)
    fig.add_trace(rw_dot, row=1, col=2)         # idx 6
    fig.add_trace(la_dot, row=1, col=2)         # idx 7

    # Frame construction
    frames = []
    for i in range(len(df)):
        (xsL, ysL), (xsR, ysR), (xsC, ysC) = frame_segments(i)
        px, py, txt = frame_points(i)
        label = f"{t[i]:.3f}s" if i < len(t) and np.isfinite(t[i]) else str(i)

        rwi = rw_x[i] if i < len(rw_x) and np.isfinite(rw_x[i]) else None
        lai = la_x[i] if i < len(la_x) and np.isfinite(la_x[i]) else None
        ti  = t[i]    if i < len(t)    and np.isfinite(t[i])    else None

        frames.append(go.Frame(
            data=[
                go.Scatter(x=xsL, y=ysL, mode="lines", line=dict(color=COLORS["left"])),
                go.Scatter(x=xsR, y=ysR, mode="lines", line=dict(color=COLORS["right"])),
                go.Scatter(x=xsC, y=ysC, mode="lines", line=dict(color=COLORS["center"])),
                go.Scatter(x=px, y=py, mode="markers", hovertext=txt),
                go.Scatter(x=[ti], y=[rwi], mode="markers", marker=dict(color=COLORS["right"])),
                go.Scatter(x=[ti], y=[lai], mode="markers", marker=dict(color=COLORS["left"])),
            ],
            traces=[0,1,2,3,6,7],
            name=label
        ))

    steps = [dict(method="animate",
                  args=[[f.name], {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}],
                  label=f.name) for f in frames]

    updatemenus = [dict(
        type="buttons",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame":{"duration":duration_ms,"redraw":True},
                              "fromcurrent":True, "transition":{"duration":0}}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}]),
        ],
        direction="left",
        pad={"r":10,"t":60},
        x=0.1, y=1.15, xanchor="left", yanchor="top"
    )]

    sliders = [dict(
        active=0,
        steps=steps,
        x=0.1, y=1.10, xanchor="left", yanchor="top",
        len=0.8,
        pad={"t":40, "b":10}
    )]

    # Layout
    fig.update_xaxes(title_text="x", range=list(xr), scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="y", range=list(yr), autorange="reversed", row=1, col=1)

    tfinite = t[np.isfinite(t)]
    tmin, tmax = (np.nanmin(tfinite), np.nanmax(tfinite)) if tfinite.size else (0, max(1, len(df)-1))
    fig.update_xaxes(title_text="time (s)", range=[tmin, tmax], row=1, col=2)
    fig.update_yaxes(title_text="x position", row=1, col=2)

    fig.update_layout(
        margin=dict(l=20, r=20, t=100, b=20),
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=True,
        height=600,
        title_text=""
    )

    fig.frames = frames
    pio.renderers.default = "browser"
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_skeleton_2d.py <metrics.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
