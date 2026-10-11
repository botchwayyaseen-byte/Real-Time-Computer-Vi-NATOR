# Live Narrator

A real-time computer vision demo that watches a webcam, tracks people and objects, graphs body movement live, and describes what it sees out loud.

## Scripts

| File | What it does |
|------|--------------|
| `live_combined.py` | Everything together: object tracking + pose estimation + live per-joint waveforms + spoken narration |
| `live_narrator.py` | Object detection/tracking with spoken narration ("A cup enters from the right") |
| `live_body_graph.py` | Pose estimation with 13 scrolling waveforms, one per body part |

## Features

- Object detection and tracking with YOLOv8
- Pose estimation (13 joints) with scrolling waveforms for vertical movement, horizontal movement, or speed
- Movement normalised by torso length, so graphs are comparable at any distance from the camera
- Real-time oscillation detection (e.g. "Right wrist moving side to side, about 1.8 hertz")
- Spoken narration in a background thread; stale lines are dropped so the voice never falls behind
- Optional CSV export of normalised joint coordinates

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Model weights download automatically on first run.

## Run

```
python live_combined.py
python live_combined.py --no-objects        # faster, body only
python live_combined.py --conf 0.4          # fewer false object detections
python live_combined.py --csv recording.csv # save joint coordinates
```

Keys (click the video window first): `1` / `2` / `3` waveform mode, `+` / `-` zoom, `m` mute, `o` objects on/off, `q` quit.

The word used for the tracked human can be changed with the `SUBJECT` constant near the top of the scripts.

## Notes

- Voice: on Windows the built-in system voice is used through PowerShell. On macOS/Linux it falls back to `pyttsx3`.
- Running two models on a CPU is demanding; use `--no-objects`, `--obj-every`, or `--imgsz 320` if the frame rate is low.
- Only the largest person in frame is tracked for the body graphs.
