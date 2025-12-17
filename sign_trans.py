#!/usr/bin/env python3
"""
sign_trans.py
Sign-language translator with kNN classifier.
Supports Raspberry Pi Camera Module 3 via libcamera (Picamera2).

Modes:
 - collect: capture labeled examples
 - train:   train kNN model
 - infer:   live translation

Key accuracy fix:
 - MediaPipe "no hand detected" now returns None (NOT zeros)
 - collect: skips frames with no hand detected (prevents label contamination)
 - infer: shows NULL when no hand detected (optional label)

Other features kept:
 - --pi-camera (Picamera2) and --cam (USB index)
 - --swap-rb to fix blue-tint PiCamera preview
 - append prompt: "File exists. Append to it? [Y/n]:"
"""

import argparse
import time
import pickle
import os
from collections import Counter, deque

import numpy as np
import cv2

# ---------------- Camera backends ----------------

try:
    from picamera2 import Picamera2
    _HAS_PICAMERA = True
except Exception:
    _HAS_PICAMERA = False

# ---------------- MediaPipe ----------------

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
    mp_hands = mp.solutions.hands
except Exception:
    _HAS_MEDIAPIPE = False


# ---------------- Camera abstraction ----------------

class Camera:
    """Unified camera interface (Pi Camera Module 3 or USB webcam)."""
    def __init__(self, use_pi_camera: bool = False, width: int = 640, height: int = 480, cam_index: int = 0):
        self.use_pi_camera = use_pi_camera
        self.width = width
        self.height = height
        self.cam_index = int(cam_index)
        self.cam = None

    def open(self):
        if self.use_pi_camera:
            if not _HAS_PICAMERA:
                raise RuntimeError(
                    "Picamera2 not installed. Run:\n"
                    "sudo apt install python3-picamera2"
                )
            self.cam = Picamera2()
            cfg = self.cam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            self.cam.configure(cfg)
            self.cam.start()
        else:
            self.cam = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cam.isOpened():
                raise RuntimeError(f"Cannot open USB camera index {self.cam_index}")

    def read(self):
        if self.use_pi_camera:
            return True, self.cam.capture_array()
        return self.cam.read()

    def release(self):
        if self.cam is None:
            return
        if self.use_pi_camera:
            self.cam.close()
        else:
            self.cam.release()


# ---------------- Feature extractor ----------------

class FeatureExtractor:
    """
    MediaPipe hand landmarks (63 floats) or fallback grayscale vector.
    IMPORTANT: When using MediaPipe, returns None if no hand detected.
    """
    def __init__(self, use_mediapipe: bool = True, fallback_size=(64, 64)):
        self.use_mediapipe = bool(use_mediapipe and _HAS_MEDIAPIPE)
        self.fallback_size = tuple(fallback_size)
        self.feature_dim = 63 if self.use_mediapipe else int(self.fallback_size[0] * self.fallback_size[1])

        if self.use_mediapipe:
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def extract(self, frame_bgr: np.ndarray):
        if self.use_mediapipe:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._hands.process(rgb)

            if not res.multi_hand_landmarks:
                return None  # <-- critical fix

            lm = res.multi_hand_landmarks[0].landmark
            arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).reshape(-1)  # 63

            # wrist-normalize to reduce translation effects
            wrist = arr[:3].copy()
            arr = arr - np.tile(wrist, 21)

            if arr.size != 63:
                return None

            return arr.astype(np.float32)

        # fallback mode: always returns a vector (no detection concept)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, self.fallback_size, interpolation=cv2.INTER_AREA)
        vec = (small.astype(np.float32).ravel() / 255.0)
        return vec


# ---------------- kNN classifier ----------------

class KNNClassifier:
    def __init__(self, k=3):
        self.k = max(1, int(k))
        self.X = None
        self.y = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"Training X must be 2D (N,D). Got {X.shape}")

        self.X = X
        self.y = y
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-8
        self.X = (self.X - self.mean) / self.std

    def predict_one(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        x = (x - self.mean) / self.std
        dists = np.linalg.norm(self.X - x, axis=1)
        k = min(self.k, dists.size)
        idx = np.argpartition(dists, k - 1)[:k]
        return Counter(self.y[idx]).most_common(1)[0][0]


# ---------------- Data collection ----------------

def _maybe_swap_rb(frame: np.ndarray, swap_rb: bool) -> np.ndarray:
    # On some PiCamera2 setups, frames look blue-tinted; swapping fixes it.
    # We only do this if user asks (swap_rb True).
    if not swap_rb:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def collect_examples(
    label: str,
    count: int,
    out_path: str,
    extractor: FeatureExtractor,
    use_pi_camera: bool,
    cam_index: int,
    swap_rb: bool,
    delay: float,
):
    # Append prompt
    if os.path.exists(out_path):
        resp = input(f"File '{out_path}' exists. Append to it? [Y/n]: ").strip().lower()
        if resp in ("n", "no"):
            base, ext = os.path.splitext(out_path)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"{base}_{ts}{ext}"
            print(f"Using new file: {out_path}")

    cam = Camera(use_pi_camera=use_pi_camera, cam_index=cam_index)
    cam.open()

    X, y = [], []
    print(f"Collecting {count} samples for '{label}' (press q to stop)")
    time.sleep(0.4)

    try:
        while len(X) < count:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame = _maybe_swap_rb(frame, swap_rb)

            feat = extractor.extract(frame)

            # MediaPipe mode: skip if no hand detected (prevents poisoned labels)
            if extractor.use_mediapipe and feat is None:
                disp = frame.copy()
                cv2.putText(
                    disp,
                    "No hand detected - NOT saved",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    disp,
                    f"{label} {len(X)}/{count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Collect", disp)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            if feat.size != extractor.feature_dim:
                continue

            X.append(feat)
            y.append(label)

            disp = frame.copy()
            cv2.putText(
                disp,
                f"{label} {len(X)}/{count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Collect", disp)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            if delay > 0:
                time.sleep(delay)

    finally:
        cam.release()
        cv2.destroyAllWindows()

    if not X:
        print("No samples collected (all frames may have had no detected hand).")
        return

    X_new = np.stack(X, axis=0).astype(np.float32)
    y_new = np.asarray(y)

    # Append existing data (but guard feature_dim)
    if os.path.exists(out_path):
        old = np.load(out_path, allow_pickle=True)
        if "feature_dim" in old and int(old["feature_dim"]) != int(extractor.feature_dim):
            raise RuntimeError(
                f"Feature mismatch: file feature_dim={int(old['feature_dim'])} vs current={extractor.feature_dim}.\n"
                f"Collect with the same mode (mediapipe vs fallback) or use a new output file."
            )
        X_all = np.vstack([old["X"], X_new])
        y_all = np.concatenate([old["y"], y_new])
    else:
        X_all, y_all = X_new, y_new

    np.savez_compressed(out_path, X=X_all, y=y_all, feature_dim=np.array(extractor.feature_dim, dtype=np.int32))
    print(f"Saved {len(y_all)} samples to {out_path}")


# ---------------- Training ----------------

def train_model(data_path: str, model_path: str, k: int):
    d = np.load(data_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["y"])
    feature_dim = int(d["feature_dim"]) if "feature_dim" in d else X.shape[1]

    if X.ndim != 2 or X.shape[1] != feature_dim:
        raise RuntimeError(f"Bad training data shape: X={X.shape}, feature_dim={feature_dim}")

    clf = KNNClassifier(k=k)
    clf.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "feature_dim": feature_dim}, f)

    print(f"Model saved to {model_path} (k={k}, samples={X.shape[0]}, feature_dim={feature_dim})")


# ---------------- Live inference ----------------

def live_translate(
    model_path: str,
    extractor: FeatureExtractor,
    use_pi_camera: bool,
    cam_index: int,
    swap_rb: bool,
    smooth: int,
    null_label: str,
):
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    clf = payload["model"]
    expected_dim = int(payload["feature_dim"])

    if extractor.feature_dim != expected_dim:
        raise RuntimeError(
            f"Model expects feature_dim={expected_dim}, but extractor has {extractor.feature_dim}.\n"
            f"Make sure you use the same mode as training (e.g., --mediapipe)."
        )

    cam = Camera(use_pi_camera=use_pi_camera, cam_index=cam_index)
    cam.open()

    preds = deque(maxlen=max(1, int(smooth)))
    print("Live inference started (press q to quit)")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame = _maybe_swap_rb(frame, swap_rb)

            feat = extractor.extract(frame)

            # MediaPipe mode: if no hand detected -> NULL
            if extractor.use_mediapipe and feat is None:
                label = null_label
                preds.clear()  # prevents stale votes from persisting through "no hand" gaps
            else:
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
                if feat.size != expected_dim:
                    continue
                preds.append(clf.predict_one(feat))
                label = Counter(preds).most_common(1)[0][0]

            disp = frame.copy()
            cv2.putText(
                disp,
                str(label),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255) if label != null_label else (200, 200, 200),
                4,
            )
            cv2.imshow("Sign Translator", disp)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser(description="Sign language translator")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--label", required=True)
    c.add_argument("--count", type=int, default=50)
    c.add_argument("--out", default="data.npz")
    c.add_argument("--mediapipe", action="store_true")
    c.add_argument("--pi-camera", action="store_true")
    c.add_argument("--cam", type=int, default=0, help="USB webcam index (ignored if --pi-camera)")
    c.add_argument("--swap-rb", action="store_true", help="Fix blue-tinted PiCamera preview by swapping R/B")
    c.add_argument("--delay", type=float, default=0.2, help="Seconds between saved samples")

    t = sub.add_parser("train")
    t.add_argument("--data", default="data.npz")
    t.add_argument("--out", default="model.pkl")
    t.add_argument("-k", type=int, default=5)

    i = sub.add_parser("infer")
    i.add_argument("--model", default="model.pkl")
    i.add_argument("--mediapipe", action="store_true")
    i.add_argument("--pi-camera", action="store_true")
    i.add_argument("--cam", type=int, default=0, help="USB webcam index (ignored if --pi-camera)")
    i.add_argument("--swap-rb", action="store_true", help="Fix blue-tinted PiCamera preview by swapping R/B")
    i.add_argument("--smooth", type=int, default=7, help="Majority-vote window")
    i.add_argument("--null-label", default="No Hand", help="Label shown when no hand is detected")

    args = p.parse_args()

    if args.cmd == "collect":
        extractor = FeatureExtractor(use_mediapipe=args.mediapipe)
        collect_examples(
            label=args.label,
            count=args.count,
            out_path=args.out,
            extractor=extractor,
            use_pi_camera=args.pi_camera,
            cam_index=args.cam,
            swap_rb=args.swap_rb,
            delay=args.delay,
        )

    elif args.cmd == "train":
        train_model(args.data, args.out, args.k)

    elif args.cmd == "infer":
        extractor = FeatureExtractor(use_mediapipe=args.mediapipe)
        live_translate(
            model_path=args.model,
            extractor=extractor,
            use_pi_camera=args.pi_camera,
            cam_index=args.cam,
            swap_rb=args.swap_rb,
            smooth=args.smooth,
            null_label=args.null_label,
        )


if __name__ == "__main__":
    main()
