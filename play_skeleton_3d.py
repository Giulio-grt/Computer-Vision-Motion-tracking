# play_skeleton_3d.py
import sys, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def main(csv_path, flip_z=False):
    if not os.path.exists(csv_path):
        raise SystemExit(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ---- expected joint base names (must match your CSV headers: <name>_{x,y,z}) ----
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

    # Keep only joints that actually exist in the CSV (robust to column mismatches)
    def cols_exist(j): return all(f"{j}_{ax}" in df.columns for ax in ("x","y","z"))
    JOINTS = [j for j in JOINTS_ALL if cols_exist(j)]
    if not JOINTS:
        raise SystemExit("No <joint>_{x,y,z} columns found in the CSV.")

    # Bones (pairs of joints) – simple MediaPipe-style body graph
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

    # Optional Z flip (if your scene looks mirrored forward/back)
    def maybe_flip(z):
        return -z if flip_z else z

    # Build arrays for global bounds (for stable camera)
    allX, allY, allZ = [], [], []
    for j in JOINTS:
        x = pd.to_numeric(df[f"{j}_x"], errors="coerce").to_numpy()
        y = pd.to_numeric(df[f"{j}_y"], errors="coerce").to_numpy()
        z = pd.to_numeric(df[f"{j}_z"], errors="coerce").to_numpy()
        if flip_z: z = -z
        allX.append(x); allY.append(y); allZ.append(z)

    allX = np.hstack(allX); allY = np.hstack(allY); allZ = np.hstack(allZ)
    finite_mask = np.isfinite(allX) & np.isfinite(allY) & np.isfinite(allZ)
    if not finite_mask.any():
        raise SystemExit("No finite 3D points found in the CSV.")

    xmin, xmax = np.nanmin(allX[finite_mask]), np.nanmax(allX[finite_mask])
    ymin, ymax = np.nanmin(allY[finite_mask]), np.nanmax(allY[finite_mask])
    zmin, zmax = np.nanmin(allZ[finite_mask]), np.nanmax(allZ[finite_mask])

    # Pad bounds a bit
    def pad(a, b, p=0.05):
        d = (b - a) or 1.0
        return a - p*d, b + p*d
    xr = pad(xmin, xmax)
    yr = pad(ymin, ymax)
    zr = pad(zmin, zmax)

    # Build frame-wise line segments for bones + joint markers
    def frame_segments(i):
        xs, ys, zs = [], [], []
        row = df.iloc[i]
        for a, b in BONES:
            if not (cols_exist(a) and cols_exist(b)):
                continue
            ax, ay, az = row[f"{a}_x"], row[f"{a}_y"], row[f"{a}_z"]
            bx, by, bz = row[f"{b}_x"], row[f"{b}_y"], row[f"{b}_z"]
            if np.all(np.isfinite([ax,ay,az,bx,by,bz])):
                xs += [ax, bx, None]
                ys += [ay, by, None]
                zs += [maybe_flip(az), maybe_flip(bz), None]
        return xs, ys, zs

    def frame_points(i):
        px, py, pz, txt = [], [], [], []
        row = df.iloc[i]
        for j in JOINTS:
            x, y, z = row[f"{j}_x"], row[f"{j}_y"], row[f"{j}_z"]
            if np.all(np.isfinite([x,y,z])):
                px.append(x); py.append(y); pz.append(maybe_flip(z)); txt.append(j)
        return px, py, pz, txt

    # Time metadata
    t = pd.to_numeric(df.get("time_s", pd.Series(index=df.index, dtype=float)), errors="coerce")
    t = t.to_numpy()
    # Use median dt for animation speed; fallback to 30 fps
    dts = np.diff(t[np.isfinite(t)])
    duration_ms = int(max(1, np.median(dts)*1000)) if dts.size else 33

    # Create initial traces
    xs0, ys0, zs0 = frame_segments(0)
    px0, py0, pz0, txt0 = frame_points(0)

    bones_tr = go.Scatter3d(
        x=xs0, y=ys0, z=zs0,
        mode="lines",
        line=dict(width=6),
        name="bones",
        hoverinfo="skip",
        showlegend=False,
    )
    points_tr = go.Scatter3d(
        x=px0, y=py0, z=pz0,
        mode="markers+text",
        marker=dict(size=3),
        text=None,  # set to txt0 if you want labels on joints
        name="joints",
        showlegend=False,
        hovertext=txt0,
        hoverinfo="text",
    )

    # Build frames
    frames = []
    for i in range(len(df)):
        xs, ys, zs = frame_segments(i)
        px, py, pz, txt = frame_points(i)
        # Use t[i] as label if available
        label = f"{t[i]:.3f}s" if i < len(t) and np.isfinite(t[i]) else str(i)
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=xs, y=ys, z=zs),
                go.Scatter3d(x=px, y=py, z=pz, hovertext=txt),
            ],
            name=label
        ))

    # Slider steps
    steps = [dict(method="animate",
                  args=[[f.name], {"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}],
                  label=f.name) for f in frames]

    # Layout with Play/Pause
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
            title="3D Pose Replay",
            scene=dict(
                xaxis=dict(title="x (m)", range=list(xr), zeroline=False, autorange=False),
                yaxis=dict(title="y (m)", range=list(yr), zeroline=False, autorange=False),
                zaxis=dict(title="z (m)", range=list(zr), zeroline=False, autorange=False),
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, t=80, b=0),
            updatemenus=updatemenus,
            sliders=sliders,
        ),
        frames=frames
    )


    fig.show()

if __name__ == "__main__":
    # Usage: python play_skeleton_3d.py metrics_3d.csv [--flipz]
    if len(sys.argv) < 2:
        print("Usage: python play_skeleton_3d.py <metrics_3d.csv> [--flipz]")
        sys.exit(1)
    csv_path = sys.argv[1]
    flip = ("--flipz" in sys.argv[2:]) or ("-fz" in sys.argv[2:])
    main(csv_path, flip_z=flip)
