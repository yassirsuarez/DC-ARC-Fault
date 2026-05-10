#!/usr/bin/env python3
"""
generate_rocket_h_final.py
==========================
Genera rocket_weights.h e rocket_transform.h CORRETTI
basati sul C-mirror validato.

USO:
    python generate_rocket_h_final.py bundle.pkl --out ./deploy_stm32
"""

import argparse
import json
import os
import pickle
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
# Aggiungila subito dopo le righe "warnings.filterwarnings("ignore")" e "INDICES = ..."

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
INDICES = np.array(
    [list(c) for c in combinations(range(9), 3)],
    dtype=np.int32
)


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# EXPORT rocket_weights.h
# =============================================================================
def export_weights(tr, out_dir, input_len=250):

    p0      = tr.parameter
    dil0    = np.array(p0[0], dtype=np.int32)
    n_fpd0  = np.array(p0[1], dtype=np.int32)
    bias0   = np.array(p0[2], dtype=np.float32)
    n_feat0 = int(84 * n_fpd0.sum())

    p1      = tr.parameter1
    dil1    = np.array(p1[0], dtype=np.int32)
    n_fpd1  = np.array(p1[1], dtype=np.int32)
    bias1   = np.array(p1[2], dtype=np.float32)
    n_feat1 = int(84 * n_fpd1.sum())

    n_fpk   = int(tr.n_features_per_kernel)
    total   = (n_feat0 + n_feat1) * n_fpk
    n_fpt   = total // 2

    lines = []
    lines.append("/* rocket_weights.h - AUTO-GENERATED, DO NOT EDIT */")
    lines.append("/* Validato: C-mirror Python == aeon nativo 10/10  */")
    lines.append("#ifndef ROCKET_WEIGHTS_H")
    lines.append("#define ROCKET_WEIGHTS_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define ROCKET_INPUT_LEN             {input_len}")
    lines.append(f"#define ROCKET_INPUT_LEN_DIFF        {input_len - 1}")
    lines.append(f"#define ROCKET_N_KERNELS             84")
    lines.append(f"#define ROCKET_N_FEATURES_PER_KERNEL {n_fpk}")
    lines.append(f"#define ROCKET_N_FEAT0               {n_feat0}")
    lines.append(f"#define ROCKET_N_FEAT1               {n_feat1}")
    lines.append(f"#define ROCKET_N_FPT                 {n_fpt}")
    lines.append(f"#define ROCKET_TOTAL_FEATURES        {total}")
    lines.append("")

    # Indici fissi 84 kernel
    lines.append("/* Indici fissi: combinations(range(9), 3) */")
    lines.append("static const int32_t rocket_indices[84][3] = {")
    for ki in range(84):
        i0, i1, i2 = INDICES[ki]
        lines.append(f"    {{{i0}, {i1}, {i2}}},")
    lines.append("};")
    lines.append("")

    def write_set(s, dil, n_fpd, bias, n_feat):
        n_dil = len(dil)
        lines.append(f"/* === SET {s} === */")
        lines.append(f"#define ROCKET_S{s}_N_DILATIONS {n_dil}")
        lines.append(f"#define ROCKET_S{s}_N_FEATURES  {n_feat}")
        lines.append("")

        lines.append(f"static const int32_t rocket_s{s}_dilations[{n_dil}] = {{")
        lines.append("    " + ", ".join(str(v) for v in dil) + ",")
        lines.append("};")
        lines.append("")

        lines.append(f"static const int32_t rocket_s{s}_n_fpd[{n_dil}] = {{")
        lines.append("    " + ", ".join(str(v) for v in n_fpd) + ",")
        lines.append("};")
        lines.append("")

        lines.append(f"static const float rocket_s{s}_biases[{n_feat}] = {{")
        for i in range(0, n_feat, 8):
            chunk = bias[i:i+8]
            lines.append("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",")
        lines.append("};")
        lines.append("")

    write_set(0, dil0, n_fpd0, bias0, n_feat0)
    write_set(1, dil1, n_fpd1, bias1, n_feat1)

    lines.append("#endif /* ROCKET_WEIGHTS_H */")

    path = os.path.join(out_dir, "rocket_weights.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved:", path)

    return total, n_feat0, n_feat1, n_fpt


# =============================================================================
# EXPORT rocket_transform.h
# =============================================================================
def export_transform(total, n_feat0, n_feat1, n_fpt, input_len, out_dir):

    code = f"""/* rocket_transform.h - AUTO-GENERATED, DO NOT EDIT        */
/* Implementazione validata: C-mirror == aeon nativo 10/10  */
/* MultiRocket transform per STM32H7                        */
/*                                                          */
/* Pipeline su MCU:                                         */
/*   rocket_transform(adc_buf, feat_buf)                    */
/*   scaler_transform(feat_buf, ROCKET_TOTAL_FEATURES)      */
/*   pca_transform(feat_buf, pca_out, tmp_buf)              */
/*   ai_run(&net, pca_out, result)                          */
#ifndef ROCKET_TRANSFORM_H
#define ROCKET_TRANSFORM_H

#include <stdint.h>
#include <string.h>
#include "rocket_weights.h"

/* ------------------------------------------------------------------ */
/* Buffers — metti in DTCM (.dtcm_bss) per velocita' massima su H7    */
/* ------------------------------------------------------------------ */
static float _rocket_feat[ROCKET_TOTAL_FEATURES];
static float _rocket_diff[ROCKET_INPUT_LEN_DIFF];

/* ------------------------------------------------------------------ */
/* FILL ONE SET                                                         */
/*                                                                     */
/* Replica esatta di _transform_uni (aeon numba):                      */
/*   - C_alpha/C_gamma con shift vettoriali                            */
/*   - _padding1 = (_padding0 + kernel_index) % 2                     */
/*   - 4 pooling: PPV, LSPV, MPV, MIPV                                */
/*   - layout interleaved: [PPV][LSPV][MPV][MIPV]                     */
/*   - n_tp_loop = n_timepoints originale (anche per serie diff)       */
/* ------------------------------------------------------------------ */
static inline void _rocket_fill_set(
    const float*   __restrict__ x_in,
    int             n_tp,
    int             n_tp_loop,
    const int32_t* dilations,
    const int32_t* n_fpd,
    int             n_dilations,
    const float*   biases,
    int             n_features,
    float*         __restrict__ out,
    int             offset_base)
{{
    int feature_index_start = 0;

    for (int d_idx = 0; d_idx < n_dilations; d_idx++)
    {{
        int _padding0 = d_idx % 2;
        int dilation  = dilations[d_idx];
        int n_feat    = n_fpd[d_idx];
        int padding   = ((9 - 1) * dilation) / 2;

        /* C_alpha = -x, C_gamma[4] = 3*x */
        float C_alpha[{input_len}];
        float C_gamma[9][{input_len}];

        for (int t = 0; t < n_tp; t++)
        {{
            C_alpha[t] = -x_in[t];
            for (int g = 0; g < 9; g++) C_gamma[g][t] = 0.0f;
            C_gamma[4][t] = x_in[t] * 3.0f;
        }}

        /* Shift vettoriali — usa n_tp_loop come aeon */
        int start = dilation;
        int end   = n_tp_loop - padding;

        for (int gi = 0; gi < 4; gi++)
        {{
            int e = end < n_tp ? end : n_tp;
            for (int t = 0; t < e; t++)
            {{
                C_alpha[n_tp - e + t] += -x_in[t];
                C_gamma[gi][n_tp - e + t] = x_in[t] * 3.0f;
            }}
            end += dilation;
        }}

        for (int gi = 5; gi < 9; gi++)
        {{
            int s = start < n_tp ? start : n_tp;
            for (int t = s; t < n_tp; t++)
            {{
                C_alpha[t - s] += -x_in[t];
                C_gamma[gi][t - s] = x_in[t] * 3.0f;
            }}
            start += dilation;
        }}

        for (int k_idx = 0; k_idx < ROCKET_N_KERNELS; k_idx++)
        {{
            int feature_index_end = feature_index_start + n_feat;
            int _padding1 = (_padding0 + k_idx) % 2;

            int i0 = rocket_indices[k_idx][0];
            int i1 = rocket_indices[k_idx][1];
            int i2 = rocket_indices[k_idx][2];

            /* C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2] */
            float C[{input_len}];
            for (int t = 0; t < n_tp; t++)
                C[t] = C_alpha[t] + C_gamma[i0][t]
                                  + C_gamma[i1][t]
                                  + C_gamma[i2][t];

            /* Seleziona porzione con o senza padding */
            const float* C_vec;
            int n_c;
            if (_padding1 == 0)
            {{
                C_vec = C;
                n_c   = n_tp;
            }}
            else
            {{
                C_vec = C + padding;
                n_c   = n_tp - 2 * padding;
            }}

            for (int feat_count = 0; feat_count < n_feat; feat_count++)
            {{
                int   feature_index = feature_index_start + feat_count;
                float bias          = biases[feature_index];

                int   ppv        = 0;
                int   last_val   = 0;
                float max_stretch = 0.0f;
                int   mean_index = 0;
                float mean       = 0.0f;

                for (int j = 0; j < n_c; j++)
                {{
                    if (C_vec[j] > bias)
                    {{
                        ppv++;
                        mean_index += j;
                        mean       += C_vec[j] + bias;
                    }}
                    else if (C_vec[j] < bias)
                    {{
                        float stretch = (float)(j - last_val);
                        if (stretch > max_stretch) max_stretch = stretch;
                        last_val = j;
                    }}
                }}
                float stretch = (float)(n_c - 1 - last_val);
                if (stretch > max_stretch) max_stretch = stretch;

                float ppv_norm = (n_c > 0) ? (float)ppv / n_c : 0.0f;
                float mpv      = (ppv > 0)  ? mean / ppv       : 0.0f;
                float mipv     = (ppv > 0)  ? (float)mean_index / ppv : -1.0f;

                int fi = feature_index + offset_base;
                out[fi]                    = ppv_norm;
                out[fi + n_features]       = max_stretch;
                out[fi + 2 * n_features]   = mpv;
                out[fi + 3 * n_features]   = mipv;
            }}

            feature_index_start = feature_index_end;
        }}
    }}
}}

/* ================================================================== */
/* API PUBBLICA                                                        */
/* ================================================================== */

/**
 * rocket_transform()
 *
 * @param x         serie float32, lunghezza ROCKET_INPUT_LEN (250)
 * @param feat_out  output, lunghezza ROCKET_TOTAL_FEATURES ({total})
 *
 * Uso tipico:
 *   rocket_transform(adc_buffer, _rocket_feat);
 */
static inline void rocket_transform(
    const float* __restrict__ x,
    float*       __restrict__ feat_out)
{{
    /* Set 0: X raw */
    _rocket_fill_set(
        x,
        ROCKET_INPUT_LEN,
        ROCKET_INPUT_LEN,           /* n_tp_loop = n_tp */
        rocket_s0_dilations,
        rocket_s0_n_fpd,
        ROCKET_S0_N_DILATIONS,
        rocket_s0_biases,
        ROCKET_N_FEAT0,
        feat_out,
        0                           /* offset_base = 0 */
    );

    /* Set 1: diff(X, 1) */
    for (int i = 0; i < ROCKET_INPUT_LEN_DIFF; i++)
        _rocket_diff[i] = x[i + 1] - x[i];

    _rocket_fill_set(
        _rocket_diff,
        ROCKET_INPUT_LEN_DIFF,
        ROCKET_INPUT_LEN,           /* n_tp_loop = 250, non 249! */
        rocket_s1_dilations,
        rocket_s1_n_fpd,
        ROCKET_S1_N_DILATIONS,
        rocket_s1_biases,
        ROCKET_N_FEAT1,
        feat_out,
        ROCKET_N_FPT                /* offset_base = n_features_per_transform */
    );
}}

#endif /* ROCKET_TRANSFORM_H */
"""

    path = os.path.join(out_dir, "rocket_transform.h")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Saved:", path)


# =============================================================================
# EXPORT main_example.c
# =============================================================================
def export_main_example(out_dir):

    code = """/* main_example.c - STM32H7 inference pipeline */
#include "rocket_transform.h"
#include "scaler.h"
#include "pca.h"

/* Colloca i buffer pesanti in RAM veloce */
__attribute__((section(".dtcm_bss")))
static float adc_buffer[ROCKET_INPUT_LEN];

__attribute__((section(".dtcm_bss")))
static float feat_buffer[ROCKET_TOTAL_FEATURES];

__attribute__((section(".dtcm_bss")))
static float pca_output[PCA_N_COMPONENTS];

__attribute__((section(".dtcm_bss")))
static float tmp_buffer[ROCKET_TOTAL_FEATURES];

/* Soglia decisione (default 0.5, calibra sul tuo dataset) */
#define ARC_THRESHOLD 0.5f

void arc_detection_run(void)
{
    /* 1. Acquisisci campioni ADC in adc_buffer[] */
    /* HAL_ADC_Start_DMA(...); */
    /* ... normalizza raw -> float ... */

    /* 2. MultiRocket transform */
    rocket_transform(adc_buffer, feat_buffer);

    /* 3. StandardScaler */
    scaler_transform(feat_buffer, ROCKET_TOTAL_FEATURES);

    /* 4. PCA */
    pca_transform(feat_buffer, pca_output, tmp_buffer);

    /* 5. ST Edge AI (ArcNet) */
    /* ai_run(&arcnet, pca_output, result); */

    /* 6. Decisione */
    /* float prob = sigmoid(result[0]); */
    /* if (prob > ARC_THRESHOLD) arc_detected(); */
}
"""

    path = os.path.join(out_dir, "main_example.c")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Saved:", path)


# =============================================================================
# MAIN
# =============================================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--out",       default="./deploy_stm32")
    parser.add_argument("--input-len", type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("LOAD")
    print("=" * 60)
    bundle = load_bundle(args.bundle)
    tr     = bundle["transformer"]
    print("Transformer:", type(tr).__name__)

    print()
    print("=" * 60)
    print("EXPORT WEIGHTS")
    print("=" * 60)
    total, n_feat0, n_feat1, n_fpt = export_weights(tr, args.out, args.input_len)

    print()
    print("=" * 60)
    print("EXPORT TRANSFORM")
    print("=" * 60)
    export_transform(total, n_feat0, n_feat1, n_fpt, args.input_len, args.out)

    print()
    print("=" * 60)
    print("EXPORT MAIN EXAMPLE")
    print("=" * 60)
    export_main_example(args.out)

    # Stima memoria
    bias_kb  = (n_feat0 + n_feat1) * 4 / 1024
    feat_kb  = total * 4 / 1024
    stack_kb = args.input_len * 10 * 4 / 1024  # C_alpha + C_gamma stack

    cfg = {
        "input_len":    args.input_len,
        "n_feat0":      n_feat0,
        "n_feat1":      n_feat1,
        "total_features": total,
        "n_fpt":        n_fpt,
        "flash_kb":     round(bias_kb, 1),
        "feat_ram_kb":  round(feat_kb, 1),
        "stack_kb":     round(stack_kb, 1),
    }

    with open(os.path.join(args.out, "rocket_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print()
    print("=" * 60)
    print("MEMORY ESTIMATE STM32H7")
    print("=" * 60)
    print(f"  Biases (Flash int)    : {bias_kb:.1f} KB")
    print(f"  Feature buffer (RAM)  : {feat_kb:.1f} KB")
    print(f"  Stack per transform   : ~{stack_kb:.1f} KB")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"""
Output in: {args.out}/
    rocket_weights.h   -> biases + indici (Flash interna)
    rocket_transform.h -> transform validato inline in C
    main_example.c     -> esempio pipeline completa STM32H7

Step 4 — STM32CubeIDE:
    1. Copia tutti i .h in Core/Inc/
    2. Carica arcnet.onnx su ST Edge AI Developer Cloud
    3. Genera codice C e importa in CubeIDE
    4. Integra main_example.c con HAL ADC
""")


if __name__ == "__main__":
    main()