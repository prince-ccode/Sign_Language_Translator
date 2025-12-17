#!/usr/bin/env python3
import argparse, glob, os
import numpy as np

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y = d["y"]
    feat = int(d["feature_dim"]) if "feature_dim" in d else (X.shape[1] if getattr(X, "ndim", 0) == 2 else None)
    return X, y, feat

def coerce_2d(X, feature_dim):
    # Handle object/ragged arrays safely
    if isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype != object:
        if X.shape[1] != feature_dim:
            raise ValueError(f"Expected dim {feature_dim}, got {X.shape[1]}")
        return X.astype(np.float32, copy=False)

    rows = []
    for r in list(X):
        rr = np.asarray(r).reshape(-1)
        if rr.size == feature_dim:
            rows.append(rr.astype(np.float32))
    if not rows:
        return np.zeros((0, feature_dim), dtype=np.float32)
    return np.stack(rows, axis=0)

def expand_inputs(inputs):
    files = []
    for item in inputs:
        # If exact file exists, use it; otherwise treat it as a glob
        if os.path.exists(item):
            files.append(item)
        else:
            files.extend(glob.glob(item))
    # de-dup + stable order
    seen = set()
    out = []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def main():
    ap = argparse.ArgumentParser(description="Merge multiple .npz datasets into one (X,y,feature_dim).")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help='One or more file paths and/or glob patterns (e.g. "Letter_*.npz" "Null.npz")')
    ap.add_argument("--out", default="data.npz", help="Output merged dataset")
    args = ap.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit(f"No files matched inputs: {args.inputs}")

    X_all, y_all = [], []
    feature_dim = None

    for f in files:
        X, y, feat = load_npz(f)

        if feat is None:
            raise SystemExit(f"Could not infer feature_dim from {f} (missing feature_dim and X not 2D).")

        if feature_dim is None:
            feature_dim = int(feat)

        if int(feat) != int(feature_dim):
            raise SystemExit(f"Feature mismatch in {f}: expected {feature_dim}, got {feat}")

        X2 = coerce_2d(X, feature_dim)
        y2 = np.asarray(y)

        if X2.shape[0] != y2.shape[0]:
            raise SystemExit(f"Row mismatch in {f}: X rows={X2.shape[0]} vs y rows={y2.shape[0]}")

        X_all.append(X2)
        y_all.append(y2)

        # print summary
        uniq = sorted(set(map(str, y2)))
        print(f"Loaded {f}: X={X2.shape}, labels={uniq}")

    X_merged = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, feature_dim), dtype=np.float32)
    y_merged = np.concatenate(y_all, axis=0) if y_all else np.array([], dtype=np.str_)

    np.savez_compressed(args.out, X=X_merged, y=y_merged, feature_dim=np.array(feature_dim, dtype=np.int32))
    print(f"\nSaved merged dataset: {args.out}")
    print(f"X: {X_merged.shape}")
    print(f"Classes: {sorted(set(map(str, y_merged)))}")

if __name__ == "__main__":
    main()
