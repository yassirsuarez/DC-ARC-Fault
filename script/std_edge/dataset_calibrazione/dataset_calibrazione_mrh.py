#!/usr/bin/env python3
"""
create_calibration_dataset_npz.py
=================================

Genera un dataset di calibrazione .npz per ST Edge AI / STM32Cube.AI.

INPUT:
    dataset .npz con:
        X -> shape (N, T) oppure (N, C, T)
        y -> labels

OUTPUT:
    calibration_dataset.npz
        X -> shape (N_calib, C, T)
        y -> shape (N_calib,)

    calibration_config.json

USO:
    python create_calibration_dataset_npz.py dataset.npz

ESEMPIO:
    python create_calibration_dataset_npz.py arc_dataset_test.npz \
        --n-samples 200 \
        --downsample 4 \
        --balanced \
        --out ./calibration_mrh

NOTE:
- ST Edge AI usa il dataset SOLO per quantizzazione/calibrazione INT8
- Il preprocessing DEVE essere identico al training
- Stesso downsample del training
"""

import argparse
import json
import os

import numpy as np


RANDOM_STATE = 42


# -------------------------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------------------------
def load_dataset(path, downsample_factor=4):

    data = np.load(path)

    X = data["X"]
    y = data["y"]

    # (N, T) -> (N, 1, T)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    # stesso preprocessing del training
    X = X[:, :, ::downsample_factor]

    return X.astype(np.float32), y.astype(np.int64)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")

    parser.add_argument(
        "--out",
        default="./calibration_mrh"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Numero campioni calibrazione"
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Deve essere IDENTICO al training"
    )

    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Dataset bilanciato"
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("LOAD DATASET")
    print("=" * 60)

    X, y = load_dataset(
        args.dataset,
        downsample_factor=args.downsample
    )

    print("Dataset shape:", X.shape)
    print("Labels shape :", y.shape)

    rng = np.random.default_rng(RANDOM_STATE)

    # ---------------------------------------------------------------------
    # SAMPLE SELECTION
    # ---------------------------------------------------------------------

    if args.balanced:

        print("\nBalanced calibration dataset")

        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]

        n_half = args.n_samples // 2

        sel0 = rng.choice(
            idx0,
            size=min(n_half, len(idx0)),
            replace=False
        )

        sel1 = rng.choice(
            idx1,
            size=min(n_half, len(idx1)),
            replace=False
        )

        indices = np.concatenate([sel0, sel1])

        rng.shuffle(indices)

    else:

        indices = rng.choice(
            len(X),
            size=min(args.n_samples, len(X)),
            replace=False
        )

    X_calib = X[indices]
    y_calib = y[indices]

    print("\nCalibration shape:", X_calib.shape)

    # ---------------------------------------------------------------------
    # SAVE NPZ
    # ---------------------------------------------------------------------

    out_path = os.path.join(
        args.out,
        "calibration_dataset.npz"
    )

    np.savez_compressed(
        out_path,
        X=X_calib,
        y=y_calib
    )

    print("\nSaved:")
    print(out_path)

    # ---------------------------------------------------------------------
    # CONFIG
    # ---------------------------------------------------------------------

    cfg = {
        "n_samples": int(len(X_calib)),
        "input_shape": list(X_calib.shape),
        "dtype": "float32",
        "downsample": args.downsample,
        "balanced": bool(args.balanced),
    }

    cfg_path = os.path.join(
        args.out,
        "calibration_config.json"
    )

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(cfg_path)

    # ---------------------------------------------------------------------
    # INFO
    # ---------------------------------------------------------------------

    print()
    print("=" * 60)
    print("ST EDGE AI")
    print("=" * 60)

    print(f"""
Dataset calibrazione:
    {out_path}

Contenuto:
    X -> {X_calib.shape}
    y -> {y_calib.shape}

Input model shape:
    (batch, 1, T)

Esempio:
    (200, 1, 250)

IMPORTANTE:
- preprocessing IDENTICO al training
- stesso downsample
- dtype float32
""")

    print("=" * 60)


if __name__ == "__main__":
    main()