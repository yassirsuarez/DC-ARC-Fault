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


class FeatureScaler:
    def fit(self, X):
        self.mean  = X.mean(axis=0).astype(np.float32)
        self.scale = X.std(axis=0).astype(np.float32)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mean) / self.scale).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


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

        lines.append(f"static const int32_t rocket_s{s}_dilations[{n_dil}] = {{")
        lines.append("    " + ", ".join(str(v) for v in dil) + ",")
        lines.append("};")
        lines.append("")

        lines.append(f"static const int32_t rocket_s{s}_n_fpd[{n_dil}] = {{")
        lines.append("    " + ", ".join(str(v) for v in n_fpd) + ",")
        lines.append("};")
        lines.append("")

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
#ifndef ROCKET_TRANSFORM_H
#define ROCKET_TRANSFORM_H

#include <stdint.h>
#include <string.h>
#include "rocket_weights.h"

/* Feature buffer RAM: {total_features} x 4 = {total_features*4/1024:.1f} KB */
static float _rocket_feat[ROCKET_TOTAL_FEATURES];

/* Differenza primo ordine */
static float _rocket_diff[{input_len - 1}];

static inline void _rocket_compute_diff(const float* x, int n)
{{
    for (int i = 0; i < n - 1; i++)
        _rocket_diff[i] = x[i + 1] - x[i];
}}

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

        float dots[{input_len + 100}];

        for (int k = 0; k < ROCKET_N_KERNELS; k++)
        {{
            int i0 = rocket_indices[k][0];
            int i1 = rocket_indices[k][1];
            int i2 = rocket_indices[k][2];

            for (int t = 0; t < out_len; t++)
            {{
                int p0 = t + i0 * dilation - padding;
                int p1 = t + i1 * dilation - padding;
                int p2 = t + i2 * dilation - padding;

                float v0 = (p0 >= 0 && p0 < input_len) ? x[p0] : 0.0f;
                float v1 = (p1 >= 0 && p1 < input_len) ? x[p1] : 0.0f;
                float v2 = (p2 >= 0 && p2 < input_len) ? x[p2] : 0.0f;

                dots[t] = -v0 + 2.0f * v1 - v2;
            }}

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

static inline void rocket_transform(
    const float* __restrict__ x,
    float*       __restrict__ feat_out)
{{
    int idx = 0;

    _rocket_apply_set(
        x, ROCKET_INPUT_LEN,
        rocket_s0_dilations, rocket_s0_n_fpd,
        ROCKET_S0_N_DILATIONS, rocket_s0_biases,
        feat_out, &idx
    );

    _rocket_compute_diff(x, ROCKET_INPUT_LEN);
    _rocket_apply_set(
        _rocket_diff, ROCKET_INPUT_LEN - 1,
        rocket_s1_dilations, rocket_s1_n_fpd,
        ROCKET_S1_N_DILATIONS, rocket_s1_biases,
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
# — usa concatenazione di stringhe invece di r'''...''' per evitare
#   problemi con f-string e quote dentro raw string
# =============================================================================
def export_validator(out_dir):

    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""')
    lines.append('rocket_validate.py')
    lines.append('==================')
    lines.append('Verifica C-mirror Python vs aeon nativo.')
    lines.append('')
    lines.append('USO:')
    lines.append('    python rocket_validate.py bundle.pkl test.npz')
    lines.append('"""')
    lines.append('')
    lines.append('import pickle')
    lines.append('import numpy as np')
    lines.append('from itertools import combinations')
    lines.append('import argparse')
    lines.append('import warnings')
    lines.append('warnings.filterwarnings("ignore")')
    lines.append('')
    lines.append('INDICES = np.array(')
    lines.append('    [list(c) for c in combinations(range(9), 3)],')
    lines.append('    dtype=np.int32')
    lines.append(')')
    lines.append('')
    lines.append('class FeatureScaler:')
    lines.append('    def fit(self, X):')
    lines.append('        self.mean  = X.mean(axis=0).astype(np.float32)')
    lines.append('        self.scale = X.std(axis=0).astype(np.float32)')
    lines.append('        self.scale[self.scale < 1e-8] = 1.0')
    lines.append('        return self')
    lines.append('    def transform(self, X):')
    lines.append('        return ((X - self.mean) / self.scale).astype(np.float32)')
    lines.append('    def fit_transform(self, X):')
    lines.append('        return self.fit(X).transform(X)')
    lines.append('')
    lines.append('def load_bundle(path):')
    lines.append('    with open(path, "rb") as f:')
    lines.append('        return pickle.load(f)')
    lines.append('')
    lines.append('def load_dataset(path, downsample=4):')
    lines.append('    data = np.load(path)')
    lines.append('    X = data["X"]')
    lines.append('    y = data["y"]')
    lines.append('    if X.ndim == 2:')
    lines.append('        X = X[:, np.newaxis, :]')
    lines.append('    if downsample > 1:')
    lines.append('        X = X[:, :, ::downsample]')
    lines.append('    return X.astype(np.float32), y.astype(np.int64)')
    lines.append('')
    lines.append('def rocket_transform_py(x_1d, tr):')
    lines.append('')
    lines.append('    n_timepoints = len(x_1d)')
    lines.append('')
    lines.append('    p0      = tr.parameter')
    lines.append('    dil0    = np.array(p0[0], dtype=np.int32)')
    lines.append('    n_fpd0  = np.array(p0[1], dtype=np.int32)')
    lines.append('    bias0   = np.array(p0[2], dtype=np.float32)')
    lines.append('    n_feat0 = int(84 * n_fpd0.sum())')
    lines.append('')
    lines.append('    p1      = tr.parameter1')
    lines.append('    dil1    = np.array(p1[0], dtype=np.int32)')
    lines.append('    n_fpd1  = np.array(p1[1], dtype=np.int32)')
    lines.append('    bias1   = np.array(p1[2], dtype=np.float32)')
    lines.append('    n_feat1 = int(84 * n_fpd1.sum())')
    lines.append('')
    lines.append('    n_features_per_kernel    = int(tr.n_features_per_kernel)')
    lines.append('    total                    = (n_feat0 + n_feat1) * n_features_per_kernel')
    lines.append('    n_features_per_transform = total // 2')
    lines.append('')
    lines.append('    features = np.zeros(total, dtype=np.float32)')
    lines.append('')
    lines.append('    def fill_set(x_in, dilations, n_fpd, biases, n_features, offset_base,')
    lines.append('                 n_timepoints_orig=None):')
    lines.append('        n_tp      = len(x_in)')
    lines.append('        n_tp_loop = n_timepoints_orig if n_timepoints_orig is not None else n_tp')
    lines.append('        feature_index_start = 0')
    lines.append('')
    lines.append('        for d_idx in range(len(dilations)):')
    lines.append('            _padding0 = d_idx % 2')
    lines.append('            dilation  = int(dilations[d_idx])')
    lines.append('            n_feat    = int(n_fpd[d_idx])')
    lines.append('            padding   = ((9 - 1) * dilation) // 2')
    lines.append('')
    lines.append('            A = -x_in')
    lines.append('            G = x_in * 3.0')
    lines.append('')
    lines.append('            C_alpha = np.zeros(n_tp, dtype=np.float32)')
    lines.append('            C_alpha[:] = A')
    lines.append('            C_gamma = np.zeros((9, n_tp), dtype=np.float32)')
    lines.append('            C_gamma[4] = G')
    lines.append('')
    lines.append('            start = dilation')
    lines.append('            end   = n_tp_loop - padding')
    lines.append('')
    lines.append('            for gi in range(4):')
    lines.append('                e = min(end, n_tp)')
    lines.append('                C_alpha[-e:] += A[:e]')
    lines.append('                C_gamma[gi, -e:] = G[:e]')
    lines.append('                end += dilation')
    lines.append('')
    lines.append('            for gi in range(5, 9):')
    lines.append('                s = min(start, n_tp)')
    lines.append('                C_alpha[:-s] += A[s:]')
    lines.append('                C_gamma[gi, :-s] = G[s:]')
    lines.append('                start += dilation')
    lines.append('')
    lines.append('            for k_idx in range(84):')
    lines.append('                feature_index_end = feature_index_start + n_feat')
    lines.append('                _padding1 = (_padding0 + k_idx) % 2')
    lines.append('')
    lines.append('                i0, i1, i2 = INDICES[k_idx]')
    lines.append('                C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]')
    lines.append('')
    lines.append('                C_vec = C if _padding1 == 0 else C[padding:-padding]')
    lines.append('                n_c   = len(C_vec)')
    lines.append('')
    lines.append('                for feat_count in range(n_feat):')
    lines.append('                    feature_index = feature_index_start + feat_count')
    lines.append('                    bias = float(biases[feature_index])')
    lines.append('')
    lines.append('                    ppv = last_val = 0')
    lines.append('                    max_stretch = 0.0')
    lines.append('                    mean_index = mean = 0.0')
    lines.append('')
    lines.append('                    for j in range(n_c):')
    lines.append('                        if C_vec[j] > bias:')
    lines.append('                            ppv        += 1')
    lines.append('                            mean_index += j')
    lines.append('                            mean       += float(C_vec[j]) + bias')
    lines.append('                        elif C_vec[j] < bias:')
    lines.append('                            stretch = j - last_val')
    lines.append('                            if stretch > max_stretch:')
    lines.append('                                max_stretch = stretch')
    lines.append('                            last_val = j')
    lines.append('')
    lines.append('                    stretch = n_c - 1 - last_val')
    lines.append('                    if stretch > max_stretch:')
    lines.append('                        max_stretch = stretch')
    lines.append('')
    lines.append('                    ppv_norm = float(ppv) / n_c if n_c > 0 else 0.0')
    lines.append('                    mpv      = mean / ppv              if ppv > 0 else 0.0')
    lines.append('                    mipv     = float(mean_index) / ppv if ppv > 0 else -1.0')
    lines.append('')
    lines.append('                    fi = feature_index + offset_base')
    lines.append('                    features[fi]                  = ppv_norm')
    lines.append('                    features[fi + n_features]     = max_stretch')
    lines.append('                    features[fi + 2 * n_features] = mpv')
    lines.append('                    features[fi + 3 * n_features] = mipv')
    lines.append('')
    lines.append('                feature_index_start = feature_index_end')
    lines.append('')
    lines.append('    fill_set(x_1d, dil0, n_fpd0, bias0,')
    lines.append('             n_features=n_feat0, offset_base=0,')
    lines.append('             n_timepoints_orig=None)')
    lines.append('')
    lines.append('    x_diff = np.diff(x_1d).astype(np.float32)')
    lines.append('    fill_set(x_diff, dil1, n_fpd1, bias1,')
    lines.append('             n_features=n_feat1,')
    lines.append('             offset_base=n_features_per_transform,')
    lines.append('             n_timepoints_orig=n_timepoints)')
    lines.append('')
    lines.append('    return features')
    lines.append('')
    lines.append('')
    lines.append('def main():')
    lines.append('    parser = argparse.ArgumentParser()')
    lines.append('    parser.add_argument("bundle")')
    lines.append('    parser.add_argument("test")')
    lines.append('    parser.add_argument("--downsample", type=int, default=4)')
    lines.append('    parser.add_argument("--n-samples",  type=int, default=10)')
    lines.append('    args = parser.parse_args()')
    lines.append('')
    lines.append('    bundle = load_bundle(args.bundle)')
    lines.append('    tr     = bundle["transformer"]')
    lines.append('    X, y   = load_dataset(args.test, args.downsample)')
    lines.append('')
    lines.append('    print("Transform nativo aeon...")')
    lines.append('    F_native = tr.transform(X[:args.n_samples]).astype(np.float32)')
    lines.append('    print(f"Native shape: {F_native.shape}")')
    lines.append('')
    lines.append('    print("\\nC-mirror Python...")')
    lines.append('    errors = []')
    lines.append('')
    lines.append('    h1, h2, h3, h4 = "idx", "max_diff", "mean_diff", "match"')
    lines.append('    print(f"\\n{h1:>4}  {h2:>12}  {h3:>12}  {h4:>6}")')
    lines.append('    print("-" * 50)')
    lines.append('')
    lines.append('    for i in range(args.n_samples):')
    lines.append('        x_1d  = X[i, 0, :]')
    lines.append('        f_c   = rocket_transform_py(x_1d, tr)')
    lines.append('        f_nat = F_native[i]')
    lines.append('        n     = min(len(f_c), len(f_nat))')
    lines.append('        diff  = np.abs(f_c[:n] - f_nat[:n])')
    lines.append('        max_d  = diff.max()')
    lines.append('        mean_d = diff.mean()')
    lines.append('        match  = max_d < 1e-3')
    lines.append('        if not match:')
    lines.append('            errors.append(i)')
    lines.append('        ok_str = "OK" if match else "WARN"')
    lines.append('        print(f"{i:>4}  {max_d:>12.6f}  {mean_d:>12.6f}  {ok_str:>6}")')
    lines.append('')
    lines.append('    print("-" * 50)')
    lines.append('    print(f"Match: {len(errors.__class__.__name__) and args.n_samples - len(errors)}/{args.n_samples}")')
    lines.append('    print(f"Match: {args.n_samples - len(errors)}/{args.n_samples}")')
    lines.append('')
    lines.append('    if not errors:')
    lines.append('        print("\\nVALIDATION PASSED!")')
    lines.append('        print("Il C-mirror e corretto — pronti per generare rocket_transform.h finale")')
    lines.append('    else:')
    lines.append('        f_c   = rocket_transform_py(X[0, 0, :], tr)')
    lines.append('        f_nat = F_native[0]')
    lines.append('        diff  = np.abs(f_c - f_nat)')
    lines.append('        worst = np.argsort(diff)[-10:]')
    lines.append('        print("\\nPeggiori 10 indici:")')
    lines.append('        for idx in sorted(worst):')
    lines.append('            print(f"  [{idx:6d}] c={f_c[idx]:.6f}  nat={f_nat[idx]:.6f}  diff={diff[idx]:.6f}")')
    lines.append('')
    lines.append('        n_feat = 6216')
    lines.append('        print("\\nDiff per zona:")')
    lines.append('        for nome, s, e in [')
    lines.append('            ("PPV  set0", 0,          n_feat),')
    lines.append('            ("LSPV set0", n_feat,     2*n_feat),')
    lines.append('            ("MPV  set0", 2*n_feat,   3*n_feat),')
    lines.append('            ("MIPV set0", 3*n_feat,   4*n_feat),')
    lines.append('            ("PPV  set1", 4*n_feat,   5*n_feat),')
    lines.append('            ("LSPV set1", 5*n_feat,   6*n_feat),')
    lines.append('            ("MPV  set1", 6*n_feat,   7*n_feat),')
    lines.append('            ("MIPV set1", 7*n_feat,   8*n_feat),')
    lines.append('        ]:')
    lines.append('            z = diff[s:e]')
    lines.append('            print(f"  {nome}: max={z.max():.4f} mean={z.mean():.6f} n_errors={np.sum(z>1e-3)}")')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    main()')

    code = "\n".join(lines)

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

    print("=" * 60)
    print("LOAD BUNDLE")
    print("=" * 60)
    bundle = load_bundle(args.bundle)
    tr     = bundle["transformer"]
    print("Transformer:", type(tr).__name__)

    print()
    print("=" * 60)
    print("INSPECT TRANSFORMER")
    print("=" * 60)
    inspect_transformer(tr)

    print()
    print("=" * 60)
    print("EXTRACT PARAMS")
    print("=" * 60)
    indices, params = extract_params(tr)

    print()
    print("=" * 60)
    print("EXPORT WEIGHTS HEADER")
    print("=" * 60)
    total_features = export_weights_header(indices, params, args.out, args.input_len)
    print(f"Total features: {total_features}")

    print()
    print("=" * 60)
    print("EXPORT TRANSFORM HEADER")
    print("=" * 60)
    export_transform_header(params, total_features, args.out, args.input_len)

    print()
    print("=" * 60)
    print("EXPORT VALIDATOR")
    print("=" * 60)
    export_validator(args.out)

    bias_kb = sum(p["total_features"] * 4 for p in params) / 1024
    feat_kb = total_features * 4 / 1024
    idx_kb  = 84 * 3 * 4 / 1024

    cfg = {
        "n_sets":         len(params),
        "total_features": total_features,
        "input_len":      args.input_len,
        "flash_kb":       round(bias_kb + idx_kb, 1),
        "sets": [
            {
                "name":        p["name"],
                "n_dilations": p["n_dilations"],
                "n_features":  p["total_features"],
                "dilations":   p["dilations"].tolist(),
                "n_fpd":       p["n_features_per_dilation"].tolist(),
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