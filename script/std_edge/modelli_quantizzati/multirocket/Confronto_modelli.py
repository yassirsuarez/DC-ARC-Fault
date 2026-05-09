#!/usr/bin/env python3
"""
validate_mrh_c.py
=================
Valida che il runtime C (mrh_runtime.h) produca le stesse
predizioni del modello Python su campioni reali.

Genera:
    validation_samples.h   -> campioni C da usare nel test su STM32
    validation_report.json -> report completo

USO:
    python validate_mrh_c.py mrh_model.pkl test.npz
"""

import argparse
import json
import os
import pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore")


# =============================================================================
# LOAD
# =============================================================================
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(path, downsample=4):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    X = X[:, :, ::downsample]
    return X.astype(np.float32), y.astype(np.int64)


# =============================================================================
# REIMPLEMENTA IL TRANSFORM IN PYTHON (specchio esatto del C)
# Questo ci dice se la nostra implementazione C e' corretta
# =============================================================================
def extract_components(model):
    tr  = model._transform_multirocket
    sc  = model._scale_multirocket
    clf = model.classifier

    return {
        "dil0":      tr.parameter[0].astype(np.int32),
        "pad0":      tr.parameter[1].astype(np.int32),
        "w0":        tr.parameter[2].astype(np.float32),
        "dil1":      tr.parameter1[0].astype(np.int32),
        "pad1":      tr.parameter1[1].astype(np.int32),
        "w1":        tr.parameter1[2].astype(np.float32),
        "indices":   tr._indices.astype(np.int32),
        "mean":      sc.mean_.astype(np.float32),
        "scale":     sc.scale_.astype(np.float32),
        "coef":      clf.coef_.flatten().astype(np.float32),
        "intercept": float(clf.intercept_[0]),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_kernels_py(x, dils, pads, ws, n_kernels, kernel_len=9):
    """
    Reimplementazione Python IDENTICA al C in mrh_runtime.h
    PPV + max + mean + mean_above_median per ogni kernel
    """
    features = []
    w_offset = 0

    for k in range(n_kernels):
        dil = int(dils[k])
        pad = int(pads[k])
        kw  = ws[w_offset : w_offset + kernel_len]

        input_len = len(x)
        out_len   = input_len + 2 * pad - dil * (kernel_len - 1)

        dots = []
        for i in range(out_len):
            dot = 0.0
            for j in range(kernel_len):
                idx = i + j * dil - pad
                xi  = x[idx] if 0 <= idx < input_len else 0.0
                dot += kw[j] * xi
            dots.append(dot)

        dots = np.array(dots, dtype=np.float32)

        ppv  = float(np.mean(dots > 0))
        maxv = float(np.max(dots))
        mean = float(np.mean(dots))
        mam  = float(np.mean(dots[dots > 0])) if np.any(dots > 0) else 0.0

        features.extend([ppv, maxv, mean, mam])
        w_offset += kernel_len

    return np.array(features, dtype=np.float32)


def apply_hydra_py(x, indices):
    """
    Reimplementazione Python IDENTICA al C in mrh_runtime.h
    """
    features = []
    input_len = len(x)

    for i in range(len(indices)):
        ka  = int(indices[i, 0])
        kb  = int(indices[i, 1])
        lag = int(indices[i, 2])

        diff_ppv = 0.0
        count    = 0

        for t in range(lag, input_len):
            da = x[t]     - x[t - lag]
            db = x[t - 1] - x[t - lag]
            if da > db:
                diff_ppv += 1.0
            count += 1

        ppv_val  = diff_ppv / count if count > 0 else 0.0
        hash_val = float(ka ^ kb)
        lag_norm = float(lag) / input_len
        mean_idx = float(ka + kb) / 2.0

        features.extend([ppv_val, hash_val, lag_norm, mean_idx])

    return np.array(features, dtype=np.float32)


def predict_c_mirror(x_1d, c):
    """
    Predizione C-mirror: stessa logica di mrh_runtime.h
    x_1d: array 1D float32, lunghezza 250
    """
    # 1. MultiRocket set 0
    f0 = apply_kernels_py(x_1d, c["dil0"], c["pad0"], c["w0"], len(c["dil0"]))

    # 2. MultiRocket set 1
    f1 = apply_kernels_py(x_1d, c["dil1"], c["pad1"], c["w1"], len(c["dil1"]))

    # 3. Hydra
    fh = apply_hydra_py(x_1d, c["indices"])

    # 4. Concat
    feat = np.concatenate([f0, f1, fh])

    # 5. StandardScaler
    n = min(len(feat), len(c["mean"]))
    feat[:n] = (feat[:n] - c["mean"][:n]) / c["scale"][:n]

    # 6. Ridge
    n_coef = min(len(feat), len(c["coef"]))
    score  = c["intercept"] + np.dot(feat[:n_coef], c["coef"][:n_coef])

    return float(sigmoid(score))


# =============================================================================
# CONFRONTO Python nativo vs C-mirror
# =============================================================================
def compare(model, c, X, y, n_samples=100):

    rng      = np.random.default_rng(42)
    indices  = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    results  = []
    errors   = []
    agree    = 0

    print(f"\n{'idx':>4}  {'label':>5}  {'py_native':>10}  {'c_mirror':>10}  {'diff':>10}  {'match':>6}")
    print("-" * 55)

    for i, idx in enumerate(indices):

        x_1d = X[idx, 0, :]   # shape (250,)

        # Predizione Python nativa (ground truth)
        py_prob = float(model.predict_proba(X[idx:idx+1])[0, 1])

        # Predizione C-mirror
        c_prob  = predict_c_mirror(x_1d, c)

        diff    = abs(py_prob - c_prob)
        match   = diff < 0.01   # tolleranza 1%

        if match:
            agree += 1
        else:
            errors.append({
                "idx":      int(idx),
                "label":    int(y[idx]),
                "py_prob":  round(py_prob, 6),
                "c_prob":   round(c_prob,  6),
                "diff":     round(diff,    6),
            })

        results.append({
            "idx":     int(idx),
            "label":   int(y[idx]),
            "py_prob": round(py_prob, 6),
            "c_prob":  round(c_prob,  6),
            "diff":    round(diff,    6),
            "match":   bool(match),
        })

        marker = "OK" if match else "WARN"
        print(f"{idx:>4}  {y[idx]:>5}  {py_prob:>10.4f}  {c_prob:>10.4f}  {diff:>10.6f}  {marker:>6}")

    print("-" * 55)
    print(f"Match: {agree}/{len(indices)} ({100*agree/len(indices):.1f}%)")

    if errors:
        print(f"\nWARNING: {len(errors)} campioni con diff > 1%:")
        for e in errors[:5]:
            print(f"  idx={e['idx']} py={e['py_prob']:.4f} c={e['c_prob']:.4f} diff={e['diff']:.6f}")

    return results, errors, indices


# =============================================================================
# GENERA validation_samples.h per test su STM32
# =============================================================================
def generate_validation_header(X, y, results, out_dir, n_export=10):

    lines = []
    lines.append("/* validation_samples.h — AUTO-GENERATED */")
    lines.append("#ifndef VALIDATION_SAMPLES_H")
    lines.append("#define VALIDATION_SAMPLES_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append(f"#define N_VALIDATION_SAMPLES {n_export}")
    lines.append(f"#define SAMPLE_LEN           {X.shape[2]}")
    lines.append("")

    for i, r in enumerate(results[:n_export]):

        idx   = r["idx"]
        x_1d  = X[idx, 0, :]
        label = r["label"]
        py_p  = r["py_prob"]

        lines.append(f"/* Sample {i}: idx={idx} label={label} py_prob={py_p:.6f} */")
        lines.append(f"static const float val_sample_{i}[SAMPLE_LEN] = {{")

        for j in range(0, len(x_1d), 8):
            chunk = x_1d[j:j+8]
            lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")

        lines.append("};")
        lines.append(f"static const int   val_label_{i}  = {label};")
        lines.append(f"static const float val_py_prob_{i} = {py_p:.6f}f;")
        lines.append("")

    # Array di puntatori
    lines.append(f"static const float* val_samples[{n_export}] = {{")
    lines.append("    " + ", ".join(f"val_sample_{i}" for i in range(n_export)))
    lines.append("};")
    lines.append(f"static const int val_labels[{n_export}] = {{")
    lines.append("    " + ", ".join(str(results[i]["label"]) for i in range(n_export)))
    lines.append("};")
    lines.append(f"static const float val_py_probs[{n_export}] = {{")
    lines.append("    " + ", ".join(f"{results[i]['py_prob']:.6f}f" for i in range(n_export)))
    lines.append("};")
    lines.append("")
    lines.append("#endif /* VALIDATION_SAMPLES_H */")

    path = os.path.join(out_dir, "validation_samples.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:", path)


# =============================================================================
# GENERA validation_main.c per STM32
# =============================================================================
def generate_validation_main(out_dir):

    code = """/* validation_main.c
   Copia questo in STM32CubeIDE e runnalo su H7.
   Confronta mrh_predict() con i valori Python attesi.
*/
#include <stdio.h>
#include <math.h>
#include "mrh_runtime.h"
#include "validation_samples.h"

void run_validation(void)
{
    int    pass = 0;
    float  max_diff = 0.0f;

    for (int i = 0; i < N_VALIDATION_SAMPLES; i++)
    {
        float c_prob    = mrh_predict(val_samples[i]);
        float py_prob   = val_py_probs[i];
        float diff      = fabsf(c_prob - py_prob);

        if (diff > max_diff) max_diff = diff;

        int ok = (diff < 0.01f);
        if (ok) pass++;

        /* In CubeIDE usa printf via SWO o UART */
        printf("sample %2d: label=%d py=%.4f c=%.4f diff=%.6f %s\\n",
               i, val_labels[i], py_prob, c_prob, diff,
               ok ? "OK" : "WARN");
    }

    printf("\\nResult: %d/%d pass, max_diff=%.6f\\n",
           pass, N_VALIDATION_SAMPLES, max_diff);

    if (pass == N_VALIDATION_SAMPLES)
        printf("VALIDATION PASSED\\n");
    else
        printf("VALIDATION FAILED — controlla mrh_runtime.h\\n");
}
"""

    path = os.path.join(out_dir, "validation_main.c")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Saved:", path)


# =============================================================================
# MAIN
# =============================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("test")
    parser.add_argument("--out",        default="./validation_mrh")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--n-samples",  type=int, default=100)
    parser.add_argument("--n-export",   type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -------------------------------------------------------------------------
    print("=" * 60)
    print("LOAD")
    print("=" * 60)

    bundle = load_bundle(args.bundle)
    model  = bundle["model"]
    print("Model:", type(model).__name__)

    X, y = load_dataset(args.test, args.downsample)
    print("Test:", X.shape, y.shape)

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("EXTRACT COMPONENTS")
    print("=" * 60)

    c = extract_components(model)
    print("OK")

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("COMPARE Python native vs C-mirror")
    print("=" * 60)

    results, errors, _ = compare(model, c, X, y, args.n_samples)

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("GENERATE VALIDATION FILES")
    print("=" * 60)

    generate_validation_header(X, y, results, args.out, args.n_export)
    generate_validation_main(args.out)

    # -------------------------------------------------------------------------
    report = {
        "n_samples":   args.n_samples,
        "n_match":     sum(1 for r in results if r["match"]),
        "n_warn":      len(errors),
        "match_pct":   round(100 * sum(1 for r in results if r["match"]) / len(results), 2),
        "max_diff":    round(max(r["diff"] for r in results), 6),
        "mean_diff":   round(float(np.mean([r["diff"] for r in results])), 6),
        "errors":      errors[:20],
    }

    rpath = os.path.join(args.out, "validation_report.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 60)
    print("REPORT")
    print("=" * 60)
    print(f"  Match     : {report['n_match']}/{args.n_samples} ({report['match_pct']}%)")
    print(f"  Max diff  : {report['max_diff']}")
    print(f"  Mean diff : {report['mean_diff']}")
    print()

    if report["match_pct"] == 100.0:
        print("  PERFECT — runtime C identico a Python")
    elif report["match_pct"] >= 95.0:
        print("  GOOD — differenze minime, accettabile per produzione")
    else:
        print("  WARNING — differenze significative, controllare mrh_runtime.h")
        print("  In particolare verificare:")
        print("    - mean_above_median (C usa ppv>0.5, Python usa mediana reale)")
        print("    - Hydra indices layout (ka, kb, lag)")

    print()
    print(f"  Output: {args.out}/")
    print("    validation_report.json")
    print("    validation_samples.h   -> copia in STM32CubeIDE")
    print("    validation_main.c      -> copia in STM32CubeIDE")


if __name__ == "__main__":
    main()