#!/usr/bin/env python3
"""
export_rocket_transform.py
==========================
Estrae i parametri MultiRocket dal bundle.pkl e genera:
    rocket_weights.h    — biases + indici (Flash interna)
    rocket_transform.h  — transform completo inline in C
    rocket_validate.py  — verifica Python C-mirror vs aeon

Struttura REALE MultiRocket aeon:
    parameter  = (dilations, n_features_per_dilation, biases)
    parameter1 = (dilations, n_features_per_dilation, biases)  <- su diff(X,1)
    _indices   = (84, 3)  kernel fissi combinations(range(9),3)

    Il transform NON ha pesi random — usa kernel fissi con pesi [-1, 2, -1]
    e confronta il dot product con bias (quantili appresi sul train set).

USO:
    python export_rocket_transform.py bundle.pkl --out results_mrh_nn --input-len 250
    python results_mrh_nn/rocket_validate.py bundle.pkl test.npz
"""

import argparse
import json
import os
import pickle
import numpy as np
import warnings
from itertools import combinations
warnings.filterwarnings("ignore")


# =============================================================================
# LOAD
# =============================================================================
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# INSPECT
# =============================================================================
def inspect_transformer(tr):
    print("  Type    :", type(tr).__name__)

    for pname in ["parameter", "parameter1"]:
        p = getattr(tr, pname, None)
        if p is None:
            continue
        dil   = np.array(p[0], dtype=np.int32)
        n_fpd = np.array(p[1], dtype=np.int32)
        bias  = np.array(p[2], dtype=np.float32)
        print(f"\n  {pname}:")
        print(f"    dilations               : {dil.tolist()}")
        print(f"    n_features_per_dilation : {n_fpd.tolist()}")
        print(f"    sum(n_fpd)              : {n_fpd.sum()}")
        print(f"    biases shape            : {bias.shape}")
        print(f"    expected (84*sum(n_fpd)): {84 * n_fpd.sum()}")


# =============================================================================
# EXTRACT
# =============================================================================
def extract_params(tr):
    """
    Restituisce:
        indices : (84, 3) int32  — kernel fissi
        params  : lista di dict, uno per parameter/parameter1
    """

    # Indici fissi 84 kernel
    indices = np.array(
        [list(c) for c in combinations(range(9), 3)],
        dtype=np.int32
    )  # (84, 3)

    params = []

    for pname in ["parameter", "parameter1"]:
        p = getattr(tr, pname, None)
        if p is None:
            break

        dilations               = np.array(p[0], dtype=np.int32)
        n_features_per_dilation = np.array(p[1], dtype=np.int32)
        biases                  = np.array(p[2], dtype=np.float32)

        n_dil          = len(dilations)
        n_kernels      = 84
        total_features = int(n_kernels * n_features_per_dilation.sum())

        print(f"  {pname}: {n_dil} dilations, {total_features} features")
        print(f"    biases shape: {biases.shape} — atteso: {total_features}")

        assert biases.shape[0] == total_features, \
            f"biases shape mismatch: {biases.shape[0]} != {total_features}"

        params.append({
            "name":                    pname,
            "dilations":               dilations,
            "n_features_per_dilation": n_features_per_dilation,
            "biases":                  biases,
            "n_dilations":             n_dil,
            "n_kernels":               n_kernels,
            "total_features":          total_features,
        })

    return indices, params


# =============================================================================
# EXPORT rocket_weights.h
# =============================================================================
def export_weights_header(indices, params, out_dir, input_len=250):

    total_features = sum(p["total_features"] for p in params)

    lines = []
    lines.append("/* rocket_weights.h - AUTO-GENERATED, DO NOT EDIT */")
    lines.append("#ifndef ROCKET_WEIGHTS_H")
    lines.append("#define ROCKET_WEIGHTS_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define ROCKET_INPUT_LEN      {input_len}")
    lines.append(f"#define ROCKET_N_SETS         {len(params)}")
    lines.append(f"#define ROCKET_N_KERNELS      84")
    lines.append(f"#define ROCKET_TOTAL_FEATURES {total_features}")
    lines.append("")

    # Indici fissi (84, 3)
    flat_idx = indices.flatten()
    lines.append("/* Indici fissi 84 kernel: combinations(range(9), 3) */")
    lines.append(f"static const int32_t rocket_indices[84][3] = {{")
    for ki in range(84):
        i0, i1, i2 = indices[ki]
        lines.append(f"    {{{i0}, {i1}, {i2}}},")
    lines.append("};")
    lines.append("")

    for s, p in enumerate(params):
        dil   = p["dilations"]
        n_fpd = p["n_features_per_dilation"]
        bias  = p["biases"]
        n_dil = p["n_dilations"]
        feats = p["total_features"]

        lines.append(f"/* === SET {s} ({p['name']}) === */")
        lines.append(f"#define ROCKET_S{s}_N_DILATIONS  {n_dil}")
        lines.append(f"#define ROCKET_S{s}_N_FEATURES   {feats}")
        lines.append("")

        # dilations
        lines.append(f"static const int32_t rocket_s{s}_dilations[{n_dil}] = {{")
        lines.append("    " + ", ".join(str(v) for v in dil) + ",")
        lines.append("};")
        lines.append("")

        # n_features_per_dilation
        lines.append(f"static const int32_t rocket_s{s}_n_fpd[{n_dil}] = {{")
        lines.append("    " + ", ".join(str(v) for v in n_fpd) + ",")
        lines.append("};")
        lines.append("")

        # biases — layout: [dilation0_kernel0_feat0, ..., dilation0_kernel83_featN, dilation1_...]
        lines.append(f"/* biases shape: (84 * sum(n_fpd),) = ({feats},) */")
        lines.append(f"static const float rocket_s{s}_biases[{feats}] = {{")
        for i in range(0, feats, 8):
            chunk = bias[i:i+8]
            lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
        lines.append("};")
        lines.append("")

    lines.append("#endif /* ROCKET_WEIGHTS_H */")

    path = os.path.join(out_dir, "rocket_weights.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved:", path)

    return total_features


# =============================================================================
# EXPORT rocket_transform.h
# =============================================================================
def export_transform_header(params, total_features, out_dir, input_len=250):

    code = f"""/* rocket_transform.h - AUTO-GENERATED, DO NOT EDIT */
/* MultiRocket transform per STM32H7                   */
/*                                                     */
/* Struttura:                                          */
/*   84 kernel fissi con pesi [-1, 2, -1]              */
/*   Per ogni dilation: dot product > bias -> PPV      */
/*   2 set: X raw + X differenziata (np.diff)          */
#ifndef ROCKET_TRANSFORM_H
#define ROCKET_TRANSFORM_H

#include <stdint.h>
#include <string.h>
#include "rocket_weights.h"

/* ------------------------------------------------------------------ */
/* Feature buffer RAM — {total_features} x 4 = {total_features*4/1024:.1f} KB                    */
/* Metti in DTCM (.dtcm_bss) per velocita' massima su H7              */
/* ------------------------------------------------------------------ */
static float _rocket_feat[ROCKET_TOTAL_FEATURES];

/* ------------------------------------------------------------------ */
/* DIFFERENZA PRIMA ORDINE (np.diff equivalente)                       */
/* ------------------------------------------------------------------ */
static float _rocket_diff[{input_len - 1}];

static inline void _rocket_compute_diff(const float* x, int n)
{{
    for (int i = 0; i < n - 1; i++)
        _rocket_diff[i] = x[i + 1] - x[i];
}}

/* ------------------------------------------------------------------ */
/* APPLY ONE SET                                                        */
/*                                                                     */
/* Per ogni dilation d:                                                */
/*   padding = ((9-1)*d) / 2                                           */
/*   Per ogni kernel k (84):                                           */
/*     Per ogni timestep t:                                            */
/*       dot = -x[t + i0*d] + 2*x[t + i1*d] - x[t + i2*d]           */
/*     Per ogni feature f (n_fpd[d]):                                  */
/*       PPV = mean(dot > bias[offset])                                */
/* ------------------------------------------------------------------ */
static inline void _rocket_apply_set(
    const float*   __restrict__ x,
    int             input_len,
    const int32_t* dilations,
    const int32_t* n_fpd,
    int             n_dilations,
    const float*   biases,
    float*         __restrict__ out,
    int*            out_idx)
{{
    int bias_offset = 0;

    for (int d_idx = 0; d_idx < n_dilations; d_idx++)
    {{
        int dilation = dilations[d_idx];
        int n_feat   = n_fpd[d_idx];
        int padding  = ((9 - 1) * dilation) / 2;
        int out_len  = input_len + 2 * padding - dilation * (9 - 1);

        /* Pre-calcola tutti i dot product per questo dilation */
        /* Buffer temporaneo sullo stack — max out_len ~250    */
        float dots[{input_len + 100}];

        for (int k = 0; k < ROCKET_N_KERNELS; k++)
        {{
            int i0 = rocket_indices[k][0];
            int i1 = rocket_indices[k][1];
            int i2 = rocket_indices[k][2];

            /* Calcola dot products per tutti i timestep */
            for (int t = 0; t < out_len; t++)
            {{
                int p0 = t + i0 * dilation - padding;
                int p1 = t + i1 * dilation - padding;
                int p2 = t + i2 * dilation - padding;

                float v0 = (p0 >= 0 && p0 < input_len) ? x[p0] : 0.0f;
                float v1 = (p1 >= 0 && p1 < input_len) ? x[p1] : 0.0f;
                float v2 = (p2 >= 0 && p2 < input_len) ? x[p2] : 0.0f;

                /* Kernel fisso: pesi -1, 2, -1 */
                dots[t] = -v0 + 2.0f * v1 - v2;
            }}

            /* n_feat PPV con soglie diverse (bias) */
            for (int f = 0; f < n_feat; f++)
            {{
                float bias = biases[bias_offset];
                float ppv  = 0.0f;

                for (int t = 0; t < out_len; t++)
                    if (dots[t] > bias) ppv += 1.0f;

                out[(*out_idx)++] = (out_len > 0) ? ppv / out_len : 0.0f;

                bias_offset++;
            }}
        }}
    }}
}}

/* ================================================================== */
/* API PUBBLICA                                                        */
/* ================================================================== */

/**
 * rocket_transform()
 *
 * @param x         serie float32, lunghezza ROCKET_INPUT_LEN
 * @param feat_out  output, lunghezza ROCKET_TOTAL_FEATURES
 *
 * Uso:
 *   rocket_transform(adc_buffer, _rocket_feat);
 *   scaler_transform(_rocket_feat, ROCKET_TOTAL_FEATURES);   // scaler.h
 *   pca_transform(_rocket_feat, pca_out, tmp_buf);           // pca.h
 *   ai_run(&network, pca_out, result);                       // ST Edge AI
 */
static inline void rocket_transform(
    const float* __restrict__ x,
    float*       __restrict__ feat_out)
{{
    int idx = 0;

    /* Set 0: X raw */
    _rocket_apply_set(
        x, ROCKET_INPUT_LEN,
        rocket_s0_dilations,
        rocket_s0_n_fpd,
        ROCKET_S0_N_DILATIONS,
        rocket_s0_biases,
        feat_out, &idx
    );

    /* Set 1: diff(X, 1) */
    _rocket_compute_diff(x, ROCKET_INPUT_LEN);
    _rocket_apply_set(
        _rocket_diff, ROCKET_INPUT_LEN - 1,
        rocket_s1_dilations,
        rocket_s1_n_fpd,
        ROCKET_S1_N_DILATIONS,
        rocket_s1_biases,
        feat_out, &idx
    );
}}

#endif /* ROCKET_TRANSFORM_H */
"""

    path = os.path.join(out_dir, "rocket_transform.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Saved:", path)


# =============================================================================
# EXPORT rocket_validate.py
# =============================================================================
def export_validator(out_dir):

    code = r'''#!/usr/bin/env python3
"""
rocket_validate.py
==================
Verifica C-mirror Python vs aeon nativo.

USO:
    python rocket_validate.py bundle.pkl test.npz
"""

import pickle
import numpy as np
import warnings
from itertools import combinations
warnings.filterwarnings("ignore")


INDICES = np.array(
    [list(c) for c in combinations(range(9), 3)],
    dtype=np.int32
)  # (84, 3)


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(path, downsample=4):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    if downsample > 1:
        X = X[:, :, ::downsample]
    return X.astype(np.float32), y.astype(np.int64)


def apply_set_py(x, dilations, n_fpd, biases):
    """
    C-mirror CORRETTO di _rocket_apply_set().
    Kernel fisso [-1, 2, -1], PPV con soglia bias.
    """
    features  = []
    input_len = len(x)
    bias_offset = 0

    for d_idx, dilation in enumerate(dilations):
        n_feat  = int(n_fpd[d_idx])
        padding = ((9 - 1) * dilation) // 2
        out_len = input_len + 2 * padding - dilation * (9 - 1)

        for k in range(84):
            i0, i1, i2 = INDICES[k]

            # Dot products per tutti i timestep
            dots = np.zeros(out_len, dtype=np.float32)
            for t in range(out_len):
                p0 = t + i0 * dilation - padding
                p1 = t + i1 * dilation - padding
                p2 = t + i2 * dilation - padding

                v0 = x[p0] if 0 <= p0 < input_len else 0.0
                v1 = x[p1] if 0 <= p1 < input_len else 0.0
                v2 = x[p2] if 0 <= p2 < input_len else 0.0

                dots[t] = -v0 + 2.0 * v1 - v2

            # n_feat PPV con soglie diverse
            for f in range(n_feat):
                bias = float(biases[bias_offset])
                ppv  = float(np.mean(dots > bias)) if out_len > 0 else 0.0
                features.append(ppv)
                bias_offset += 1

    return np.array(features, dtype=np.float32)


def rocket_transform_py(x_1d, tr):
    """
    Replica esatta di rocket_transform() in C.
    x_1d: array 1D float32, lunghezza input_len
    """
    # Set 0: X raw
    p0    = tr.parameter
    feat0 = apply_set_py(
        x_1d,
        np.array(p0[0], dtype=np.int32),
        np.array(p0[1], dtype=np.int32),
        np.array(p0[2], dtype=np.float32),
    )

    # Set 1: diff(X, 1)
    x_diff = np.diff(x_1d).astype(np.float32)
    p1     = tr.parameter1
    feat1  = apply_set_py(
        x_diff,
        np.array(p1[0], dtype=np.int32),
        np.array(p1[1], dtype=np.int32),
        np.array(p1[2], dtype=np.float32),
    )

    return np.concatenate([feat0, feat1])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("test")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--n-samples",  type=int, default=20)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    tr     = bundle["transformer"]
    X, y   = load_dataset(args.test, args.downsample)

    print("Transform nativo aeon (ground truth)...")
    F_native = tr.transform(X[:args.n_samples]).astype(np.float32)

    print("C-mirror Python...")
    errors = []

    print(f"\n{'idx':>4}  {'max_diff':>12}  {'mean_diff':>12}  {'match':>6}")
    print("-" * 45)

    for i in range(args.n_samples):
        x_1d  = X[i, 0, :]
        f_c   = rocket_transform_py(x_1d, tr)
        f_nat = F_native[i]

        n     = min(len(f_c), len(f_nat))
        diff  = np.abs(f_c[:n] - f_nat[:n])
        max_d = diff.max()
        mean_d = diff.mean()
        match = max_d < 1e-4

        if not match:
            errors.append(i)

        print(f"{i:>4}  {max_d:>12.8f}  {mean_d:>12.8f}  {'OK' if match else 'WARN':>6}")

    print("-" * 45)
    print(f"Match: {args.n_samples - len(errors)}/{args.n_samples}")

    if not errors:
        print("\nVALIDATION PASSED - rocket_transform.h e' corretto!")
    else:
        print(f"\nWARNING: {len(errors)} campioni con diff > 1e-4")


if __name__ == "__main__":
    main()
'''

    path = os.path.join(out_dir, "rocket_validate.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Saved:", path)


# =============================================================================
# MAIN
# =============================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--out",       default="./results_mrh_nn")
    parser.add_argument("--input-len", type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -------------------------------------------------------------------------
    print("=" * 60)
    print("LOAD BUNDLE")
    print("=" * 60)
    bundle = load_bundle(args.bundle)
    tr     = bundle["transformer"]
    print("Transformer:", type(tr).__name__)

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("INSPECT TRANSFORMER")
    print("=" * 60)
    inspect_transformer(tr)

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("EXTRACT PARAMS")
    print("=" * 60)
    indices, params = extract_params(tr)

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("EXPORT WEIGHTS HEADER")
    print("=" * 60)
    total_features = export_weights_header(indices, params, args.out, args.input_len)
    print(f"Total features: {total_features}")

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("EXPORT TRANSFORM HEADER")
    print("=" * 60)
    export_transform_header(params, total_features, args.out, args.input_len)

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("EXPORT VALIDATOR")
    print("=" * 60)
    export_validator(args.out)

    # -------------------------------------------------------------------------
    bias_kb  = sum(p["total_features"] * 4 for p in params) / 1024
    feat_kb  = total_features * 4 / 1024
    idx_kb   = 84 * 3 * 4 / 1024

    cfg = {
        "n_sets":         len(params),
        "total_features": total_features,
        "input_len":      args.input_len,
        "flash_kb":       round(bias_kb + idx_kb, 1),
        "sets": [
            {
                "name":          p["name"],
                "n_dilations":   p["n_dilations"],
                "n_features":    p["total_features"],
                "dilations":     p["dilations"].tolist(),
                "n_fpd":         p["n_features_per_dilation"].tolist(),
            }
            for p in params
        ]
    }

    with open(os.path.join(args.out, "rocket_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print()
    print("=" * 60)
    print("MEMORY ESTIMATE")
    print("=" * 60)
    print(f"  Biases + indices (Flash int) : {bias_kb + idx_kb:.1f} KB")
    print(f"  Feature buffer   (RAM)       : {feat_kb:.1f} KB")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"""
Output in: {args.out}/
    rocket_weights.h    -> biases + indici kernel (Flash interna)
    rocket_transform.h  -> transform completo inline in C
    rocket_validate.py  -> verifica C-mirror vs aeon

Prossimo step — validazione:
    python {args.out}/rocket_validate.py bundle.pkl test.npz
""")


if __name__ == "__main__":
    main()