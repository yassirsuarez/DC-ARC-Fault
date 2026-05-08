#!/usr/bin/env python3
"""
export_mrh.py
=============

Export ONNX del Ridge interno a:
    MultiRocketHydraClassifier (aeon)

INPUT:
    mrh_model.pkl

OUTPUT:
    ridge_inference.onnx
    ridge_inference.h
    export_config.json

USO:
    python export_mrh.py mrh_model.pkl
"""

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np


# =============================================================================
# FIND RIDGE (SAFE)
# =============================================================================
def find_ridge(model):

    visited = set()

    def recurse(obj, depth=0):

        if depth > 20:
            return None

        obj_id = id(obj)

        if obj_id in visited:
            return None

        visited.add(obj_id)

        # ---------------------------------------------------------
        # RIDGE FOUND
        # ---------------------------------------------------------
        try:
            coef = getattr(obj, "coef_", None)

            if coef is not None:
                if hasattr(coef, "shape"):
                    if coef.size > 0:
                        return obj

        except Exception:
            pass

        # ---------------------------------------------------------
        # LIST / TUPLE
        # ---------------------------------------------------------
        if isinstance(obj, (list, tuple)):

            for item in obj:

                # sklearn pipeline step
                if isinstance(item, tuple) and len(item) == 2:
                    item = item[1]

                result = recurse(item, depth + 1)

                if result is not None:
                    return result

            return None

        # ---------------------------------------------------------
        # DICT
        # ---------------------------------------------------------
        if isinstance(obj, dict):

            for val in obj.values():

                result = recurse(val, depth + 1)

                if result is not None:
                    return result

            return None

        # ---------------------------------------------------------
        # NORMAL OBJECT
        # ---------------------------------------------------------
        try:
            attrs = dir(obj)
        except Exception:
            return None

        for attr in attrs:

            if attr.startswith("__"):
                continue

            try:
                val = getattr(obj, attr)
            except Exception:
                continue

            # skip primitive
            if isinstance(
                val,
                (
                    int,
                    float,
                    str,
                    bool,
                    bytes,
                    bytearray,
                    type(None),
                ),
            ):
                continue

            # skip callable
            try:
                if callable(val):
                    continue
            except Exception:
                pass

            result = recurse(val, depth + 1)

            if result is not None:
                return result

        return None

    return recurse(model)


# =============================================================================
# EXPORT ONNX
# =============================================================================
def export_onnx(ridge, out_dir):

    import onnx
    from onnx import helper
    from onnx import TensorProto
    from onnx import numpy_helper

    coef = ridge.coef_.astype(np.float32)

    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    intercept = ridge.intercept_.astype(np.float32)

    n_features = coef.shape[1]

    print("\nFeatures:", n_features)

    # -------------------------------------------------------------------------
    # INITIALIZERS
    # -------------------------------------------------------------------------
    coef_init = numpy_helper.from_array(
        coef.T,
        name="W"
    )

    bias_init = numpy_helper.from_array(
        intercept.reshape(1).astype(np.float32),
        name="B"
    )

    # -------------------------------------------------------------------------
    # INPUT / OUTPUT
    # -------------------------------------------------------------------------
    X = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [None, n_features]
    )

    Y = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [None, 1]
    )

    # -------------------------------------------------------------------------
    # NODES
    # -------------------------------------------------------------------------
    matmul = helper.make_node(
        "MatMul",
        ["input", "W"],
        ["matmul_out"]
    )

    add = helper.make_node(
        "Add",
        ["matmul_out", "B"],
        ["logits"]
    )

    sigmoid = helper.make_node(
        "Sigmoid",
        ["logits"],
        ["output"]
    )

    # -------------------------------------------------------------------------
    # GRAPH
    # -------------------------------------------------------------------------
    graph = helper.make_graph(
        [matmul, add, sigmoid],
        "ridge_classifier",
        [X],
        [Y],
        [coef_init, bias_init]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)]
    )

    onnx.checker.check_model(model)

    out_path = os.path.join(out_dir, "ridge_inference.onnx")

    with open(out_path, "wb") as f:
        f.write(model.SerializeToString())

    print("\nSaved:", out_path)

    return n_features


# =============================================================================
# EXPORT C HEADER
# =============================================================================
def export_header(ridge, out_dir):

    coef = ridge.coef_.flatten().astype(np.float32)
    intercept = float(ridge.intercept_[0])

    n_features = len(coef)

    lines = []

    lines.append("#ifndef RIDGE_INFERENCE_H")
    lines.append("#define RIDGE_INFERENCE_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("#include <math.h>")
    lines.append("")
    lines.append(f"#define N_FEATURES {n_features}")
    lines.append("")

    # -------------------------------------------------------------------------
    # COEFFICIENTS
    # -------------------------------------------------------------------------
    lines.append(
        f"static const float ridge_coef[{n_features}] = {{"
    )

    for i in range(0, n_features, 8):

        chunk = coef[i:i+8]

        row = ", ".join(
            [f"{x:.8f}f" for x in chunk]
        )

        lines.append("    " + row + ",")

    lines.append("};")
    lines.append("")

    lines.append(
        f"static const float ridge_intercept = {intercept:.8f}f;"
    )

    lines.append("")

    lines.append("""
static inline float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float ridge_predict_proba(const float* x)
{
    float s = ridge_intercept;

    for (int i = 0; i < N_FEATURES; i++)
    {
        s += x[i] * ridge_coef[i];
    }

    return sigmoid_f(s);
}
""")

    lines.append("#endif")

    out_path = os.path.join(out_dir, "ridge_inference.h")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print("Saved:", out_path)


# =============================================================================
# MAIN
# =============================================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("bundle")

    parser.add_argument(
        "--out",
        default="./export_mrh"
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # =========================================================================
    # LOAD
    # =========================================================================
    print("=" * 60)
    print("LOAD BUNDLE")
    print("=" * 60)

    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]

    print("Model:", type(model).__name__)

    # =========================================================================
    # FIND RIDGE
    # =========================================================================
    print()
    print("=" * 60)
    print("FIND RIDGE")
    print("=" * 60)

    ridge = find_ridge(model)

    if ridge is None:
        print("\nERROR: Ridge non trovato")
        sys.exit(1)

    print("Ridge:", type(ridge).__name__)
    print("coef shape:", ridge.coef_.shape)

    # =========================================================================
    # EXPORT ONNX
    # =========================================================================
    print()
    print("=" * 60)
    print("EXPORT ONNX")
    print("=" * 60)

    n_features = export_onnx(
        ridge,
        args.out
    )

    # =========================================================================
    # EXPORT HEADER
    # =========================================================================
    print()
    print("=" * 60)
    print("EXPORT HEADER")
    print("=" * 60)

    export_header(
        ridge,
        args.out
    )

    # =========================================================================
    # CONFIG
    # =========================================================================
    cfg = {
        "n_features": int(n_features),
        "input_shape": [1, n_features],
        "output_shape": [1, 1],
    }

    cfg_path = os.path.join(
        args.out,
        "export_config.json"
    )

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print("\nSaved:", cfg_path)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()