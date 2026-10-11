"""
Live Body Graph: webcam -> YOLOv8 pose -> scrolling waveform for every tracked body part.

Same environment as live_narrator.py (ultralytics + opencv-python). Run:
    python live_body_graph.py
    python live_body_graph.py --cam 1 --csv recording.csv

Keys (click the window first):
    1 = vertical movement   2 = horizontal movement   3 = speed
    + / - = zoom the waveforms in / out
    q = quit

Try it: wave one hand and watch the wrist (and elbow) traces turn into sine-like waves,
while the rest stay flat. Step side to side or bounce on your knees to see the others.
"""
import argparse, colorsys, csv, math, time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# (label, COCO keypoint index). L/R are the person's own left/right; the video is mirrored
# like a bathroom mirror, so "L" parts appear on the left of the screen.
PARTS = [("Nose", 0),
         ("L Shoulder", 5), ("R Shoulder", 6),
         ("L Elbow", 7), ("R Elbow", 8),
         ("L Wrist", 9), ("R Wrist", 10),
         ("L Hip", 11), ("R Hip", 12),
         ("L Knee", 13), ("R Knee", 14),
         ("L Ankle", 15), ("R Ankle", 16)]
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
MODES = ["Vertical movement", "Horizontal movement", "Speed"]
WINDOW = 8.0      # seconds of history shown
CONF = 0.4        # minimum keypoint confidence


def palette():
    """One hue per body region; the right side is a paler version of the left."""
    cols = []
    for label, _ in PARTS:
        group = {"Nose": 0, "Shoulder": 1, "Elbow": 2, "Wrist": 3, "Hip": 4, "Knee": 5, "Ankle": 6}[label.split()[-1]]
        sat = 0.45 if label.startswith("R ") else 0.9
        r, g, b = colorsys.hsv_to_rgb(group / 7.0, sat, 1.0)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


def draw_panel(H, PW, bufs, colors, now, mode, gain):
    p = np.full((H, PW, 3), (24, 24, 24), np.uint8)
    n = len(PARTS)
    sh = H / n
    label_w, right_pad = 92, 8
    px0, pw = label_w, PW - label_w - right_pad

    for k in range(int(now - WINDOW) + 1, int(now) + 1):          # scrolling 1-second grid
        x = px0 + int((k - (now - WINDOW)) / WINDOW * pw)
        cv2.line(p, (x, 0), (x, H), (38, 38, 38), 1)

    for j in range(n):
        y0, y1 = int(j * sh), int((j + 1) * sh)
        h = y1 - y0
        cv2.line(p, (0, y1), (PW, y1), (55, 55, 55), 1)
        if mode < 2:
            cv2.line(p, (px0, y0 + h // 2), (px0 + pw, y0 + h // 2), (50, 50, 50), 1)   # zero line
        cv2.putText(p, PARTS[j][0], (6, y0 + h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors[j], 1, cv2.LINE_AA)

        runs = [[]]
        for t, v in bufs[j]:
            if now - t > WINDOW:
                continue
            if v != v:                                              # NaN = part not visible
                if runs[-1]:
                    runs.append([])
                continue
            x = px0 + int((t - (now - WINDOW)) / WINDOW * pw)
            if mode < 2:                                            # centred, +/-0.5 torso lengths
                y = y0 + h / 2 - (v * gain / 0.5) * (h / 2 - 3)
            else:                                                   # 0..2 torso lengths per second
                y = y1 - 3 - (v * gain / 2.0) * (h - 6)
            runs[-1].append((x, int(min(max(y, y0 + 2), y1 - 2))))
        for run in runs:
            if len(run) > 1:
                cv2.polylines(p, [np.array(run, np.int32)], False, colors[j], 2, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--model", default="yolov8n-pose.pt", help="yolov8s-pose.pt = more accurate, slower")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--csv", default=None, help="optional: save normalised keypoint coordinates to this CSV")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.cam)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not open camera {args.cam}. Try --cam 1.")
    H, W = frame.shape[:2]
    PW = 560
    colors = palette()
    bufs = [deque(maxlen=1500) for _ in PARTS]
    state = [None] * len(PARTS)
    scale = None                       # torso length in pixels (makes the graphs body-size independent)
    mode, gain = 0, 1.0
    fps, t_prev, t_start = 0.0, time.monotonic(), time.monotonic()

    csv_file = csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["t_seconds"] + [f"{n}_{a}" for n, _ in PARTS for a in ("x", "y")])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        dt = max(now - t_prev, 1e-3)
        t_prev = now
        fps = 0.9 * fps + 0.1 / dt

        r = model(frame, imgsz=args.imgsz, conf=0.4, verbose=False)[0]      # pose on the RAW frame
        disp = cv2.flip(frame, 1)                                           # mirror only for display
        kp = None
        if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
            b = r.boxes.xyxy.cpu().numpy()
            i = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax())   # biggest person only
            xy = r.keypoints.xy[i].cpu().numpy().copy()
            cf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
            raw_norm = xy / np.array([W, H])
            xy[:, 0] = W - xy[:, 0]
            kp = (xy, cf, raw_norm)

        vals = [float("nan")] * len(PARTS)
        if kp:
            xy, cf, raw_norm = kp
            if cf[[5, 6, 11, 12]].min() > CONF:
                s = float(np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2))
                if s > 10:
                    scale = s if scale is None else scale + (s - scale) * (1 - math.exp(-dt / 0.5))
            sc = scale or H * 0.25
            a_s = 1 - math.exp(-dt / 0.06)     # light smoothing of keypoint jitter
            a_e = 1 - math.exp(-dt / 1.5)      # slow average = the "rest position" we subtract
            a_v = 1 - math.exp(-dt / 0.12)     # speed smoothing
            for j, (_, k) in enumerate(PARTS):
                if cf[k] < CONF:
                    state[j] = None            # lost: break the line, re-initialise when it returns
                    continue
                x, y = xy[k]
                S = state[j]
                if S is None:
                    S = state[j] = dict(sx=x, sy=y, ex=x, ey=y, sp=0.0)
                else:
                    px, py = S["sx"], S["sy"]
                    S["sx"] += (x - S["sx"]) * a_s
                    S["sy"] += (y - S["sy"]) * a_s
                    S["ex"] += (S["sx"] - S["ex"]) * a_e
                    S["ey"] += (S["sy"] - S["ey"]) * a_e
                    S["sp"] += (math.hypot(S["sx"] - px, S["sy"] - py) / dt - S["sp"]) * a_v
                if mode == 0:
                    vals[j] = -(S["sy"] - S["ey"]) / sc        # up is positive
                elif mode == 1:
                    vals[j] = (S["sx"] - S["ex"]) / sc         # screen-right is positive
                else:
                    vals[j] = S["sp"] / sc
            if csv_writer:
                row = [round(now - t_start, 3)]
                for _, k in PARTS:
                    row += [round(float(raw_norm[k][0]), 4), round(float(raw_norm[k][1]), 4)]
                csv_writer.writerow(row)
        else:
            state = [None] * len(PARTS)

        for j in range(len(PARTS)):
            bufs[j].append((now, vals[j]))

        # skeleton + coloured joints on the mirrored video
        if kp:
            xy, cf, _ = kp
            for a, c in EDGES:
                if cf[a] > CONF and cf[c] > CONF:
                    cv2.line(disp, tuple(map(int, xy[a])), tuple(map(int, xy[c])), (210, 210, 210), 2, cv2.LINE_AA)
            for j, (_, k) in enumerate(PARTS):
                if cf[k] > CONF:
                    cv2.circle(disp, tuple(map(int, xy[k])), 7, colors[j], -1, cv2.LINE_AA)
        else:
            cv2.putText(disp, "No person detected", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(disp, f"{MODES[mode]}  x{gain:.2f}  |  {fps:4.1f} FPS", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(disp, "1/2/3 mode   +/- zoom   q quit", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        panel = draw_panel(H, PW, bufs, colors, now, mode, gain)
        cv2.imshow("Live Body Graph", np.hstack([disp, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("1"), ord("2"), ord("3")):
            mode = key - ord("1")
            for bf in bufs:
                bf.clear()
        elif key in (ord("+"), ord("=")):
            gain = min(gain * 1.25, 8.0)
        elif key in (ord("-"), ord("_")):
            gain = max(gain / 1.25, 0.2)

    cap.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
