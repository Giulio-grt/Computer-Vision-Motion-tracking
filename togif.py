from moviepy import VideoFileClip

clip = VideoFileClip("annotated.mp4")   # optional: take only first 10 seconds
clip.write_gif("demo.gif", fps=30)