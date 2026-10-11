"""
Live Narrator: webcam -> YOLOv8 object tracking -> spoken description of what's happening.

Setup (in the VS Code terminal):
    python -m venv .venv
    .venv\\Scripts\\activate          (Windows)   |   source .venv/bin/activate   (macOS/Linux)
    pip install ultralytics opencv-python pyttsx3

Run:
    python live_narrator.py
    python live_narrator.py --cam 1 --model yolov8s.pt

Keys (click the video window first):  q = quit   m = mute/unmute voice
"""
import argparse, subprocess, sys, math, queue, threading, time
from collections import Counter, deque

import cv2
from ultralytics import YOLO


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
PLURALS = {"Subject": "people"}


def plural(label, n):
    if n == 1:
        return f"a {label}"
    return f"{n} {PLURALS.get(label, label + 's')}"


def side(x, y):
    if x < 0.25: return "left"
    if x > 0.75: return "right"
    if y < 0.25: return "top"
    if y > 0.75: return "bottom"
    return "centre"


def classify(hist):
    """hist: deque of (t, x, y, area), coordinates normalised to 0..1.
    Looks at roughly the last second of movement."""
    now = hist[-1][0]
    h = [p for p in hist if now - p[0] <= 1.0]
    if len(h) < 5 or h[-1][0] - h[0][0] < 0.5:
        return None
    t0, x0, y0, a0 = h[0]
    t1, x1, y1, a1 = h[-1]
    dx, dy = x1 - x0, y1 - y0
    speed = math.hypot(dx, dy) / max(t1 - t0, 1e-3)    # frame-widths per second
    if speed < 0.06:
        if a0 > 0 and (a1 - a0) / a0 > 0.35: return "coming closer"
        if a0 > 0 and (a1 - a0) / a0 < -0.30: return "moving away"
        return "still"
    parts = []
    if abs(dx) >= abs(dy) * 0.6: parts.append("right" if dx > 0 else "left")
    if abs(dy) >= abs(dx) * 0.6: parts.append("down" if dy > 0 else "up")
    return "moving " + " and ".join(parts) + (" quickly" if speed > 0.5 else "")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt = fastest, yolov8s.pt = more accurate")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=480, help="smaller = faster")
    ap.add_argument("--summary-every", type=float, default=15.0, help="seconds of quiet before a scene summary")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.cam}. Try --cam 1.")
    voice = Speaker()

    tracks, label_ids = {}, {}
    caption, caption_t = "Starting...", 0.0
    last_say = time.monotonic()
    fps, t_prev = 0.0, time.monotonic()
    first_frame = True

    def name_of(tid):
        lab = tracks[tid]["label"]
        ids = label_ids[lab]
        return lab if len(ids) == 1 else f"{lab} {ids.index(tid) + 1}"

    def say(text):
        nonlocal caption, caption_t, last_say
        caption, caption_t, last_say = text, time.monotonic(), time.monotonic()
        print(text)
        voice.say(text)

    say("Camera on. I am watching.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)               # mirror, so left/right match what you see
        H, W = frame.shape[:2]
        now = time.monotonic()

        r = model.track(frame, persist=True, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        img = r.plot()
        seen, present = set(), Counter()

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            xywh = r.boxes.xywh.cpu().numpy()
            for tid, c, (cx, cy, w, h) in zip(ids, clss, xywh):
                seen.add(tid)
                lab = r.names[c]
                present[lab] += 1
                nx, ny = cx / W, cy / H
                if tid not in tracks:
                    tracks[tid] = dict(label=lab, hist=deque(maxlen=40), last=now, dir=None, said=-99.0, gone=False)
                    label_ids.setdefault(lab, []).append(tid)
                    s = side(nx, ny)
                    if first_frame:
                        say(f"I can see a {lab} " + ("in the centre" if s == "centre" else f"on the {s}"))
                    else:
                        say(f"A {lab} appears in the centre" if s == "centre" else f"A {lab} enters from the {s}")
                T = tracks[tid]
                T["last"], T["gone"] = now, False
                T["hist"].append((now, nx, ny, (w * h) / (W * H)))

                d = classify(T["hist"])
                if d and d != T["dir"] and now - T["said"] > 2.0:
                    prev, T["dir"] = T["dir"], d
                    if d == "still":
                        if prev and prev != "still":
                            say(f"The {name_of(tid)} has stopped")
                            T["said"] = now
                    else:
                        say(f"The {name_of(tid)} is {d}")
                        T["said"] = now
        first_frame = False

        for tid, T in tracks.items():            # things that disappeared
            if not T["gone"] and tid not in seen and now - T["last"] > 1.0:
                T["gone"] = True
                _, lx, ly, _ = T["hist"][-1]
                s = side(lx, ly)
                say(f"The {name_of(tid)} left" + ("" if s == "centre" else f" to the {s}"))

        # periodic scene summary when nothing else has been said for a while
        if now - last_say > args.summary_every:
            if present:
                items = ", ".join(plural(l, n) for l, n in present.items())
                say(f"I can see {items}")
            else:
                say("I don't see anything right now")

        # HUD
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
        t_prev = now
        cv2.putText(img, f"{fps:4.1f} FPS  |  voice {'OFF' if voice.muted else 'ON'}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if now - caption_t < 4.0:
            cv2.rectangle(img, (0, H - 40), (W, H), (0, 0, 0), -1)
            cv2.putText(img, caption, (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("Live Narrator  (q = quit, m = mute)", img)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("m"):
            voice.muted = not voice.muted

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
