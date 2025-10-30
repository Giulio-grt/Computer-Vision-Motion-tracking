![Demo_1](assets/screenshot_example_2.png)


# Computer Vision for Human Motion Tracking 

The project uses MediaPipe’s Pose model to detect 2D body landmarks from video or webcam it then computes basic joint angles, and provides interactive visualization and analysis of the motion.

[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=purple)](https://jupyter.org/)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FF6F00?style=for-the-badge&logo=mediapipe&logoColor=white)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
--- 
The repo includes:
- Offline video analysis that writes an annotated video and a CSV of pose metrics.

- Live webcam pose visualization with FPS overlay.

![Demo_2](assets/Screenshot_example_1.png)

- An interactive Plotly player that replays the 2D skeleton from a CSV and shows the coordination between your right wrist and left ankle.

![data_analysis](assets/data_analysis_demo.png)
- A jupyter-notebook that calculates and plots your arm/leg coordination during running.

![graph](assets/graph.png)




## What’s Inside this repo
- `analyze_video.py` — Processes an input video and saves a copy with added joint landmarks and the a file with all the joints data metrics at any time.
- `camera.py` — Live webcam pose overlay

- `play_skeleton_2d.py` — Interactive Plotly replay of the 2D skeleton + live coordination between right wrist and left ankle.
- `analysis.ipynb` — Notebook to analyze in detail, arm/leg coordination.

- `examples/` — Example videos with joint overlay.

- `metrics/` - Metrics from the example videos.

## Python environment Setup

Choose either Conda or Python venv.

(Download anaconda or micromamba with this link)

[![Download Anaconda](https://img.shields.io/badge/Download-Anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/download)


1) clone this repo: 
```
git clone https://github.com/Giulio-grt/CV-human-motion.git
```
2) Create a python virtual environnement.
3) Install dependencies via pip inside the env:

   - run: 
```
pip install -r requirements.txt
```

## Run The Code 

**1) Analyze a video and export outputs**
- Ensure the env is active and deps installed.
- Put your input video in `assets/` or use the provided `assets/input_2.mp4`.
- Run: 
```
python analyze_video.py
```
  - Output files: `new_annotated.mp4` and `new_metrics.csv` in the repo root.

- To process a different file or change names, edit the constants at the top of `analyze_video.py:5` (`INPUT`, `OUT_VIDEO`, `OUT_CSV`).

---

**2) Replay the 2D skeleton**
- Use the CSV from step 1 or a sample CSV in the repo (e.g., `metrics_2.csv`).
- Run: 
```
python play_skeleton_2d.py metrics_2.csv
```
- This opens your default browser with an interactive player (play/pause, slider). If it doesn’t open, set the renderer to notebook or static as needed, by default the script uses `pio.renderers.default = "browser"`.

---

**3) Live webcam pose overlay**
- Grant OS permissions to your terminal/IDE.
- Run: 
```
python camera.py
```
- A resizable window appears with landmarks and FPS; press `q` to quit. If the camera does not open, try changing the index in `camera.py:9` from `0` to `1`.
---

**4) Data analysis Notebook**

- Run the `analyis.ipynb` notebook, with your own metrics, change:
```
csv_path   = "metrics/metrics_2.csv" 
```

## Troubleshooting  

- Camera not opening on macOS:
```
Check System Settings → Privacy & Security → Camera permissions for your terminal/IDE.
```



