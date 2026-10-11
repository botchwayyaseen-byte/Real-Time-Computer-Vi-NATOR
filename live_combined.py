"""
Live Combined: webcam -> pose + object tracking -> spoken narration AND live per-body-part waveforms.

Same environment as the other two scripts (ultralytics, opencv-python, pyttsx3). Run:
    python live_combined.py
    python live_combined.py --no-objects          (faster: body only)
    python live_combined.py --obj-every 5 --csv recording.csv

Keys (click the window first):
    1 / 2 / 3 = vertical / horizontal / speed waveforms     + / - = zoom waveforms
    m = mute voice     o = objects on/off     q = quit
"""
import argparse, subprocess, sys, colorsys, csv, math, queue, threading, time
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

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
MODE_IDX = {0: 2, 1: 1, 2: 3}      # buffer entries are (t, dx, dy, speed)
WINDOW, CONF = 8.0, 0.4
# What the narrator calls the tracked human. Change this one word to rename it everywhere.
SUBJECT = "subject"
PLURALS = {}


# --------------------------------------------------------------------------- voice
class Speaker(threading.Thread):
    """Speaks in a background thread; if narration piles up, only the newest line is spoken.
    Windows: uses the built-in Windows voice through PowerShell (reliable, nothing extra to install).
    Other systems: pyttsx3 with a fresh engine per sentence (reusing one engine often stops after line 1)."""

    def __init__(self, rate=175):
        super().__init__(daemon=True)
        self.q, self.muted, self.rate = queue.Queue(), False, rate
        self.start()

    def _speak_windows(self, text):
        safe = text.replace("'", "''")
        ps = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en*' } | Select-Object -First 1; "
              "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
              "$s.Rate = 1; "
              f"$s.Speak('{safe}')")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], creationflags=0x08000000, check=False)

    def _speak_pyttsx3(self, text):
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", self.rate)
        for v in eng.getProperty("voices"):
            if "english" in v.name.lower() or "en" in v.id.lower():
                eng.setProperty("voice", v.id)
                break
        eng.say(text)
        eng.runAndWait()
        eng.stop()

    def run(self):
        while True:
            text = self.q.get()
            while not self.q.empty():                 # skip stale lines, keep the newest
                text = self.q.get_nowait()
            if self.muted:
                continue
            try:
                (self._speak_windows if sys.platform == "win32" else self._speak_pyttsx3)(text)
            except Exception as e:
                print("voice error:", e)

    def say(self, text):
        self.q.put(text)


# --------------------------------------------------------------------------- helpers
def spoken(label):
    return {"L": "left", "R": "right"}.get(label[0], "") + " " + label.split()[-1].lower() \
        if label[:2] in ("L ", "R ") else label.lower()


def plural(label, n):
    return f"a {label}" if n == 1 else f"{n} {PLURALS.get(label, label + 's')}"


def side(x, y):
    if x < 0.25: return "left"
    if x > 0.75: return "right"
    if y < 0.25: return "top"
    if y > 0.75: return "bottom"
    return "centre"


def classify(hist):
    """hist: deque of (t, x, y, area), normalised. Looks at about the last second."""
    now = hist[-1][0]
    h = [p for p in hist if now - p[0] <= 1.0]
    if len(h) < 5 or h[-1][0] - h[0][0] < 0.5:
        return None
    t0, x0, y0, a0 = h[0]
    t1, x1, y1, a1 = h[-1]
    dx, dy = x1 - x0, y1 - y0
    speed = math.hypot(dx, dy) / max(t1 - t0, 1e-3)
    if speed < 0.06:
        if a0 > 0 and (a1 - a0) / a0 > 0.35: return "coming closer"
        if a0 > 0 and (a1 - a0) / a0 < -0.30: return "moving away"
        return "still"
    parts = []
    if abs(dx) >= abs(dy) * 0.6: parts.append("right" if dx > 0 else "left")
    if abs(dy) >= abs(dx) * 0.6: parts.append("down" if dy > 0 else "up")
    return "moving " + " and ".join(parts) + (" quickly" if speed > 0.5 else "")


def analyse_osc(buf, now):
    """Is this body part oscillating (waving, bouncing...)? Returns (amplitude, axis, hertz) or None."""
    pts = [e for e in buf if now - e[0] <= 2.0 and e[1] == e[1]]
    if len(pts) < 12 or pts[-1][0] - pts[0][0] < 1.2:
        return None
    a = np.array([[e[1], e[2]] for e in pts])
    dur = pts[-1][0] - pts[0][0]
    best = None
    for ax, name in ((0, "side to side"), (1, "up and down")):
        s = a[:, ax] - a[:, ax].mean()
        p2p = float(s.max() - s.min())
        if p2p < 0.25:                                   # ignore small movements (in torso lengths)
            continue
        sg = np.sign(s)
        sg = sg[sg != 0]
        cross = int((sg[1:] != sg[:-1]).sum())
        freq = cross / (2 * dur)
        if cross >= 3 and 0.5 <= freq <= 6 and (best is None or p2p > best[0]):
            best = (p2p, name, freq)
    return best


def palette():
    cols = []
    for label, _ in PARTS:
        group = {"Nose": 0, "Shoulder": 1, "Elbow": 2, "Wrist": 3, "Hip": 4, "Knee": 5, "Ankle": 6}[label.split()[-1]]
        r, g, b = colorsys.hsv_to_rgb(group / 7.0, 0.45 if label.startswith("R ") else 0.9, 1.0)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


def draw_panel(H, PW, bufs, colors, now, mode, gain):
    p = np.full((H, PW, 3), (24, 24, 24), np.uint8)
    n = len(PARTS)
    sh = H / n
    px0, pw = 92, PW - 92 - 8
    idx = MODE_IDX[mode]
    for k in range(int(now - WINDOW) + 1, int(now) + 1):
        x = px0 + int((k - (now - WINDOW)) / WINDOW * pw)
        cv2.line(p, (x, 0), (x, H), (38, 38, 38), 1)
    for j in range(n):
        y0, y1 = int(j * sh), int((j + 1) * sh)
        h = y1 - y0
        cv2.line(p, (0, y1), (PW, y1), (55, 55, 55), 1)
        if mode < 2:
            cv2.line(p, (px0, y0 + h // 2), (px0 + pw, y0 + h // 2), (50, 50, 50), 1)
        cv2.putText(p, PARTS[j][0], (6, y0 + h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors[j], 1, cv2.LINE_AA)
        runs = [[]]
        for e in bufs[j]:
            t, v = e[0], e[idx]
            if now - t > WINDOW:
                continue
            if v != v:
                if runs[-1]:
                    runs.append([])
                continue
            x = px0 + int((t - (now - WINDOW)) / WINDOW * pw)
            y = (y0 + h / 2 - (v * gain / 0.5) * (h / 2 - 3)) if mode < 2 else (y1 - 3 - (v * gain / 2.0) * (h - 6))
            runs[-1].append((x, int(min(max(y, y0 + 2), y1 - 2))))
        for run in runs:
            if len(run) > 1:
                cv2.polylines(p, [np.array(run, np.int32)], False, colors[j], 2, cv2.LINE_AA)
    return p


# --------------------------------------------------------------------------- objects
class Objects:
    """Tracks non-person objects and narrates them (the person is handled by the pose model)."""

    def __init__(self, say):
        self.say, self.tracks, self.label_ids, self.boxes, self.first = say, {}, {}, [], True

    def name_of(self, tid):
        lab = self.tracks[tid]["label"]
        ids = self.label_ids[lab]
        return lab if len(ids) == 1 else f"{lab} {ids.index(tid) + 1}"

    def update(self, r, now, W, H):
        seen, present = set(), Counter()
        self.boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            n = len(r.boxes)
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * n
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xywh = r.boxes.xywh.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, c, cf, (cx, cy, w, h), (x1, y1, x2, y2) in zip(ids, clss, confs, xywh, xyxy):
                lab = r.names[c]
                present[lab] += 1
                self.boxes.append((int(W - x2), int(y1), int(W - x1), int(y2), f"{lab} {cf:.2f}"))
                if tid is None:            # detected but not yet confirmed by the tracker: draw it, narrate later
                    continue
                seen.add(tid)
                nx, ny = (W - cx) / W, cy / H                        # mirrored, like the display
                if tid not in self.tracks:
                    self.tracks[tid] = dict(label=lab, hist=deque(maxlen=40), last=now, dir=None, said=-99.0, gone=False)
                    self.label_ids.setdefault(lab, []).append(tid)
                    s = side(nx, ny)
                    if self.first:
                        self.say(f"I can see a {lab} " + ("in the centre" if s == "centre" else f"on the {s}"))
                    else:
                        self.say(f"A {lab} appears in the centre" if s == "centre" else f"A {lab} enters from the {s}")
                T = self.tracks[tid]
                T["last"], T["gone"] = now, False
                T["hist"].append((now, nx, ny, (w * h) / (W * H)))
                d = classify(T["hist"])
                if d and d != T["dir"] and now - T["said"] > 2.0:
                    prev, T["dir"] = T["dir"], d
                    if d == "still":
                        if prev and prev != "still":
                            self.say(f"The {self.name_of(tid)} has stopped")
                            T["said"] = now
                    else:
                        self.say(f"The {self.name_of(tid)} is {d}")
                        T["said"] = now
        self.first = False
        for tid, T in self.tracks.items():
            if not T["gone"] and tid not in seen and now - T["last"] > 1.0:
                T["gone"] = True
                s = side(T["hist"][-1][1], T["hist"][-1][2])
                self.say(f"The {self.name_of(tid)} left" + ("" if s == "centre" else f" to the {s}"))
        return present


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--pose-model", default="yolov8n-pose.pt")
    ap.add_argument("--obj-model", default="yolov8n.pt")
    ap.add_argument("--no-objects", action="store_true", help="skip object detection (faster)")
    ap.add_argument("--obj-every", type=int, default=2, help="run object detection every N frames")
    ap.add_argument("--obj-imgsz", type=int, default=640, help="object detection resolution (bigger = finds smaller things)")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.30, help="object confidence threshold (lower = detects more)")
    ap.add_argument("--summary-every", type=float, default=20.0)
    ap.add_argument("--csv", default=None, help="save normalised keypoints over time to this CSV")
    args = ap.parse_args()

    pose = YOLO(args.pose_model)
    obj_model = None if args.no_objects else YOLO(args.obj_model)
    cap = cv2.VideoCapture(args.cam)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not open camera {args.cam}. Try --cam 1.")
    H, W = frame.shape[:2]
    PW = 560
    colors = palette()
    voice = Speaker()

    caption, caption_t, last_say = "Starting...", 0.0, time.monotonic()

    def say(text):
        nonlocal caption, caption_t, last_say
        caption, caption_t, last_say = text, time.monotonic(), time.monotonic()
        print(text)
        voice.say(text)

    objects = Objects(say)
    objects_on = obj_model is not None
    last_present = Counter()

    bufs = [deque(maxlen=1500) for _ in PARTS]
    state = [None] * len(PARTS)
    scale = None
    mode, gain = 0, 1.0
    person_present, person_last, person_pos = False, 0.0, (0.5, 0.5)
    person_hist, person_dir, person_said = deque(maxlen=40), None, -99.0
    raised = {"left": False, "right": False}
    osc_part = osc_axis = None
    osc_active, osc_seen, last_osc_say, last_osc_check = False, 0.0, -99.0, 0.0
    first_frame, frame_i = True, 0
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
        frame_i += 1

        # ---- pose (every frame) --------------------------------------------------
        pr = pose(frame, imgsz=args.imgsz, conf=0.4, verbose=False)[0]
        disp = cv2.flip(frame, 1)
        kp = bbox = None
        if pr.keypoints is not None and pr.boxes is not None and len(pr.boxes) > 0:
            b = pr.boxes.xyxy.cpu().numpy()
            i = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax())      # biggest person
            xy = pr.keypoints.xy[i].cpu().numpy().copy()
            cf = pr.keypoints.conf[i].cpu().numpy() if pr.keypoints.conf is not None else np.ones(17)
            raw_norm = xy / np.array([W, H])
            xy[:, 0] = W - xy[:, 0]
            x1, y1, x2, y2 = b[i]
            bbox = ((W - (x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, ((x2 - x1) * (y2 - y1)) / (W * H))
            kp = (xy, cf, raw_norm)

        # ---- objects (every N frames) -------------------------------------------
        if objects_on and frame_i % args.obj_every == 0:
            dr = obj_model.track(frame, persist=True, conf=args.conf, imgsz=args.obj_imgsz,
                                 classes=list(range(1, 80)), verbose=False)[0]
            last_present = objects.update(dr, now, W, H)

        # ---- body signals ---------------------------------------------------------
        nan = float("nan")
        entries = [(now, nan, nan, nan)] * len(PARTS)
        if kp:
            xy, cf, raw_norm = kp
            if cf[[5, 6, 11, 12]].min() > CONF:
                s = float(np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2))
                if s > 10:
                    scale = s if scale is None else scale + (s - scale) * (1 - math.exp(-dt / 0.5))
            sc = scale or H * 0.25
            a_s, a_e, a_v = (1 - math.exp(-dt / t) for t in (0.06, 1.5, 0.12))
            for j, (_, k) in enumerate(PARTS):
                if cf[k] < CONF:
                    state[j] = None
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
                entries[j] = (now, (S["sx"] - S["ex"]) / sc, -(S["sy"] - S["ey"]) / sc, S["sp"] / sc)
            if csv_writer:
                row = [round(now - t_start, 3)]
                for _, k in PARTS:
                    row += [round(float(raw_norm[k][0]), 4), round(float(raw_norm[k][1]), 4)]
                csv_writer.writerow(row)
        else:
            state = [None] * len(PARTS)
        for j in range(len(PARTS)):
            bufs[j].append(entries[j])

        # ---- narration: the person ----------------------------------------------
        if kp:
            person_last = now
            if not person_present:
                person_present = True
                person_hist.clear()
                person_dir = None
                say(f"I can see a {SUBJECT}" if first_frame else f"A {SUBJECT} appears")
            cx, cy, ar = bbox
            person_pos = (cx, cy)
            person_hist.append((now, cx, cy, ar))
            d = classify(person_hist)
            if d and d != person_dir and now - person_said > 2.0:
                prev, person_dir = person_dir, d
                if d == "still":
                    if prev and prev != "still":
                        say(f"The {SUBJECT} has stopped")
                        person_said = now
                else:
                    say(f"The {SUBJECT} is {d}")
                    person_said = now

            xy, cf, _ = kp
            sc = scale or H * 0.25
            for who, shd, wr in (("left", 5, 9), ("right", 6, 10)):       # raised hands
                if cf[shd] > CONF and cf[wr] > CONF:
                    if xy[wr][1] < xy[shd][1] - 0.15 * sc and not raised[who]:
                        raised[who] = True
                        say(f"{who.capitalize()} hand raised")
                    elif xy[wr][1] > xy[shd][1] - 0.05 * sc and raised[who]:
                        raised[who] = False
                        say(f"{who.capitalize()} hand lowered")

            if now - last_osc_check > 0.25:                              # waving / bouncing / swaying
                last_osc_check = now
                best = None
                for j in range(len(PARTS)):
                    res = analyse_osc(bufs[j], now)
                    if res and (best is None or res[0] > best[1][0]):
                        best = (j, res)
                if best:
                    j, (amp, axis, freq) = best
                    osc_seen, osc_active = now, True
                    changed = osc_part != j or osc_axis != axis
                    if (changed and now - last_osc_say > 3.0) or now - last_osc_say > 8.0:
                        say(f"{spoken(PARTS[j][0]).capitalize()} moving {axis}, about {freq:.1f} hertz")
                        last_osc_say = now
                    osc_part, osc_axis = j, axis
                elif osc_active and now - osc_seen > 1.5:
                    osc_active, osc_part, osc_axis = False, None, None
                    say("Movement stopped")
        elif person_present and now - person_last > 1.0:
            person_present = False
            raised = {"left": False, "right": False}
            s = side(*person_pos)
            say(f"The {SUBJECT} left" + ("" if s == "centre" else f" to the {s}"))
        first_frame = False

        # ---- periodic summary ------------------------------------------------------
        if now - last_say > args.summary_every:
            present = Counter(last_present if objects_on else {})
            if person_present:
                present[SUBJECT] += 1
            if present:
                say("I can see " + ", ".join(plural(l, n) for l, n in present.items()))
            else:
                say("I don't see anything right now")

        # ---- drawing -----------------------------------------------------------------
        if objects_on:
            for x1, y1, x2, y2, lab in objects.boxes:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (80, 200, 80), 2)
                cv2.putText(disp, lab, (x1 + 3, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 80), 2)
        if kp:
            xy, cf, _ = kp
            for a, c in EDGES:
                if cf[a] > CONF and cf[c] > CONF:
                    cv2.line(disp, tuple(map(int, xy[a])), tuple(map(int, xy[c])), (210, 210, 210), 2, cv2.LINE_AA)
            for j, (_, k) in enumerate(PARTS):
                if cf[k] > CONF:
                    cv2.circle(disp, tuple(map(int, xy[k])), 7, colors[j], -1, cv2.LINE_AA)
        cv2.putText(disp, f"{MODES[mode]} x{gain:.2f} | {fps:4.1f} FPS | voice {'OFF' if voice.muted else 'ON'}"
                          f" | objects {'ON' if objects_on else 'OFF'}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if now - caption_t < 4.0:
            cv2.rectangle(disp, (0, H - 40), (W, H), (0, 0, 0), -1)
            cv2.putText(disp, caption, (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        panel = draw_panel(H, PW, bufs, colors, now, mode, gain)
        cv2.imshow("Live Combined  (1/2/3 mode, +/- zoom, m mute, o objects, q quit)", np.hstack([disp, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            mode = key - ord("1")
        elif key in (ord("+"), ord("=")):
            gain = min(gain * 1.25, 8.0)
        elif key in (ord("-"), ord("_")):
            gain = max(gain / 1.25, 0.2)
        elif key == ord("m"):
            voice.muted = not voice.muted
        elif key == ord("o") and obj_model is not None:
            objects_on = not objects_on

    cap.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
