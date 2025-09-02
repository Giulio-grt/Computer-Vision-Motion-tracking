# play_skeleton_2d.py
import sys, os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

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

    def cols_exist(j): return all(f"{j}_{ax}" in df.columns for ax in ("x","y"))
    JOINTS = [j for j in JOINTS_ALL if cols_exist(j)]
    if not JOINTS:
        raise SystemExit("No <joint>_{x,y} columns found in the CSV.")

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
        xs, ys = [], []
        row = df.iloc[i]
        for a, b in BONES:
            if not (cols_exist(a) and cols_exist(b)):
                continue
            ax, ay = row[f"{a}_x"], row[f"{a}_y"]
            bx, by = row[f"{b}_x"], row[f"{b}_y"]
            if np.all(np.isfinite([ax,ay,bx,by])):
                xs += [ax, bx, None]
                ys += [ay, by, None]
        return xs, ys

    def frame_points(i):
        px, py, txt = [], [], []
        row = df.iloc[i]
        for j in JOINTS:
            x, y = row[f"{j}_x"], row[f"{j}_y"]
            if np.all(np.isfinite([x,y])):
                px.append(x); py.append(y); txt.append(j)
        return px, py, txt

    # Timing
    t = pd.to_numeric(df.get("time_s", pd.Series(index=df.index, dtype=float)), errors="coerce").to_numpy()
    dts = np.diff(t[np.isfinite(t)])
    duration_ms = int(max(1, np.median(dts)*1000)) if dts.size else 33

    # Initial frame
    xs0, ys0 = frame_segments(0)
    px0, py0, txt0 = frame_points(0)

    bones_tr = go.Scatter(
        x=xs0, y=ys0,
        mode="lines",
        line=dict(width=3),
        name="bones",
        hoverinfo="skip",
        showlegend=False,
    )
    points_tr = go.Scatter(
        x=px0, y=py0,
        mode="markers+text",
        marker=dict(size=6),
        text=None,  # set txt0 if you want labels
        name="joints",
        showlegend=False,
        hovertext=txt0,
        hoverinfo="text",
    )

    # Frames
    frames = []
    for i in range(len(df)):
        xs, ys = frame_segments(i)
        px, py, txt = frame_points(i)
        label = f"{t[i]:.3f}s" if i < len(t) and np.isfinite(t[i]) else str(i)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=xs, y=ys, mode="lines"),
                go.Scatter(x=px, y=py, mode="markers", hovertext=txt),
            ],
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
        x=0.1, y=1.05, xanchor="left", yanchor="top"
    )]

    sliders = [dict(
        active=0,
        steps=steps,
        x=0.1, y=1.0, xanchor="left", yanchor="top",
        len=0.8,
        pad={"t":40, "b":10}
    )]

    fig = go.Figure(
        data=[bones_tr, points_tr],
        layout=go.Layout(
            title="2D Pose Replay",
            xaxis=dict(title="x", range=list(xr), scaleanchor="y", scaleratio=1),
            yaxis=dict(title="y", range=list(yr), autorange="reversed"),  # flip Y for image coords
            margin=dict(l=0, r=0, t=80, b=0),
            updatemenus=updatemenus,
            sliders=sliders,
        ),
        frames=frames
    )
    
    pio.renderers.default = "browser"
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_skeleton_2d.py <metrics.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
