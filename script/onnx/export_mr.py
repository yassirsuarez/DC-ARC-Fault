#!/usr/bin/env python3
"""
Export Ridge model (.pkl) → STM32 C header
"""

import pickle
import numpy as np
import json
import os


def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    # supporto sia bundle custom che sklearn diretto
    if "model" in bundle:
        model = bundle["model"]
    else:
        model = bundle

    return model


def extract_weights(model):
    """
    RidgeClassifier / LogisticRegression
    """
    if hasattr(model, "coef_"):
        w = model.coef_.reshape(-1)
        b = model.intercept_.reshape(-1)[0]
        return w, b

    raise ValueError("Modello non supportato: manca coef_")


def export_header(w, b, out_path, name="ridge"):

    n = len(w)

    lines = []
    lines.append("/* AUTO-GENERATED FILE - DO NOT EDIT */")
    lines.append("#ifndef RIDGE_MODEL_H")
    lines.append("#define RIDGE_MODEL_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define N_FEATURES {n}")
    lines.append("")

    # weights
    lines.append(f"static const float {name}_w[N_FEATURES] = {{")

    for i in range(0, n, 8):
        chunk = w[i:i+8]
        lines.append("    " + ", ".join(f"{v:.8ff}" for v in chunk) + ",")

    lines.append("};")
    lines.append("")

    # bias
    lines.append(f"static const float {name}_b = {b:.8ff};")
    lines.append("")

    # inference function
    lines.append("static inline float ridge_predict(float *x)")
    lines.append("{")
    lines.append("    float y = ridge_b;")
    lines.append("    for (int i = 0; i < N_FEATURES; i++)")
    lines.append("        y += ridge_w[i] * x[i];")
    lines.append("    return y;")
    lines.append("}")
    lines.append("")

    lines.append("#endif")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print("✔ Saved:", out_path)


def export_config(model, config_path):

    cfg = {
        "model": "Ridge STM32",
        "n_features": int(len(model.coef_.reshape(-1))),
    }

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl")
    parser.add_argument("--out", default="./stm32_export")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading model...")
    model = load_model(args.pkl)

    print("Extracting weights...")
    w, b = extract_weights(model)

    print("Exporting C header...")
    export_header(w, b, os.path.join(args.out, "ridge_model.h"))

    print("Exporting config...")
    export_config(model, os.path.join(args.out, "config.json"))

    print("\nDONE → STM32 ready model generated")


if __name__ == "__main__":
    main()