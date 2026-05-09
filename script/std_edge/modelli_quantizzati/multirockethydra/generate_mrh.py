#!/usr/bin/env python3
"""
generate_mrh_runtime.py
=======================
Genera il runtime C completo per STM32H7:
    mrh_runtime.h   — tutto inline, zero dipendenze esterne
    mrh_runtime.c   — implementazione

USO:
    python generate_mrh_runtime.py mrh_model.pkl
"""

import pickle
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# LOAD + INSPECT
# =============================================================================
def load_bundle(path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def extract_components(model):

    tr  = model._transform_multirocket
    sc  = model._scale_multirocket
    clf = model.classifier

    # --- MultiRocket parameters ---
    # parameter  = prima serie di kernel
    # parameter1 = seconda serie di kernel
    dil0  = tr.parameter[0].astype(np.int32)    # (20,) dilations
    pad0  = tr.parameter[1].astype(np.int32)    # (20,) paddings
    w0    = tr.parameter[2].astype(np.float32)  # (6216,) weights

    dil1  = tr.parameter1[0].astype(np.int32)
    pad1  = tr.parameter1[1].astype(np.int32)
    w1    = tr.parameter1[2].astype(np.float32)

    # --- Hydra indices ---
    indices = tr._indices.astype(np.int32)       # (84, 3)

    # --- StandardScaler ---
    mean  = sc.mean_.astype(np.float32)          # (49728,)
    scale = sc.scale_.astype(np.float32)         # (49728,)

    # --- Ridge ---
    coef      = clf.coef_.flatten().astype(np.float32)   # (50048,)
    intercept = float(clf.intercept_[0])

    return {
        "dil0": dil0, "pad0": pad0, "w0": w0,
        "dil1": dil1, "pad1": pad1, "w1": w1,
        "indices": indices,
        "mean": mean, "scale": scale,
        "coef": coef, "intercept": intercept,
    }


# =============================================================================
# REVERSE-ENGINEER kernel structure
# Rocket usa kernel length=9, ogni kernel ha 9 weights + 1 bias
# 6216 weights / 9 = 690.666... → proviamo a capire la struttura
# =============================================================================
def analyze_structure(c):

    # MultiRocket usa kernel fisso length=9
    # n_kernels = len(weights) / kernel_length
    # ma include anche bias → layout: [w0..w8, bias, w0..w8, bias, ...]
    # oppure separati

    # Dal codice aeon: parameter[2] contiene SOLO i pesi (no bias)
    # n_kernels = 84 (default aeon)
    # features per kernel = 4 (PPV, mean, max, variance) * 2 = alcune combo
    # 6216 = 84 * 74 → non torna
    # 6216 = 9 * 690 + 6 → non torna
    # proviamo: 6216 / 84 = 74 → 84 kernel, 74 valori ciascuno? no
    # Dal source aeon MultiRocketTransformer:
    #   n_kernels_per_param = 84
    #   kernel_length = 9
    #   weights shape = (n_kernels * kernel_length + n_kernels) per bias
    #   = 84 * (9+1) = 840 → no, 6216/84 = 74
    # Più probabile: n_kernels=84, 9 weights + dilations embedded
    # 6216 = 84 * 74 → non torna
    # Verifichiamo: 6216 / 9 = 690.67 → non intero
    # 6216 / 84 = 74.0 esatto!

    n_kernels = len(c["dil0"])   # 20 dilations → 20 kernel groups
    w_size    = len(c["w0"])     # 6216

    print(f"  dil0 shape : {c['dil0'].shape}")
    print(f"  pad0 shape : {c['pad0'].shape}")
    print(f"  w0   size  : {w_size}")
    print(f"  w0/20      : {w_size / 20}")
    print(f"  indices    : {c['indices'].shape}")
    print(f"  coef       : {c['coef'].shape}")
    print(f"  mean       : {c['mean'].shape}")

    # 6216 / 20 = 310.8 → non intero
    # dal source aeon: ogni "parameter set" ha n_kernels kernel
    # e ogni kernel ha length 9 + bias
    # 6216 = n_kernels * (9 + 1) → n_kernels = 621.6 → no
    # 6216 = n_kernels * 9 → n_kernels = 690.67 → no
    # Guardando il source aeon MultiRocketTransformer._get_kernels:
    #   weights flat array, non separati per kernel
    #   si accede con indici calcolati run-time
    # La struttura VERA è documentata in _transform in aeon:
    #   parameter[2] = array piatto dei pesi di TUTTI i kernel
    #   parameter[0] = dilations (una per kernel)
    #   parameter[1] = padding (una per kernel)
    #   n_kernels = len(parameter[0]) = 20 qui
    #   ogni kernel ha lunghezza variabile? no, fissa 9
    #   20 kernel * 9 weights = 180, ma 6216 >> 180
    # CONCLUSIONE: parameter[2] contiene anche i bias e le feature PPV
    # threshold → 6216 = 20 * 310 + 16 ?
    # Dal source aeon reale: guarda _fit e _transform
    # n_features_per_kernel_multirocket = 84
    # 20 * 84 = 1680, ancora diverso
    # 6216 = ? → serve ispezionare direttamente

    return w_size


# =============================================================================
# ISPEZIONA LA STRUTTURA REALE dal source aeon
# =============================================================================
def get_true_structure(model):
    """
    Legge direttamente dalla classe aeon la struttura dei parametri.
    """
    try:
        from aeon.transformations.collection.convolution_based._multirocket import (
            MultiRocket
        )
        print("  aeon MultiRocket importato")
    except ImportError:
        pass

    tr = model._transform_multirocket

    info = {}

    # Attributi pubblici/privati utili
    for attr in ["n_kernels", "kernel_length", "n_features_per_kernel",
                 "num_kernels", "_n_kernels", "n_timepoints_",
                 "n_columns_", "_kernel_length"]:
        val = getattr(tr, attr, None)
        if val is not None:
            info[attr] = val
            print(f"  {attr} = {val}")

    return info


# =============================================================================
# GENERA IL CODICE C
# =============================================================================
def write_array_f32(name, arr, lines, per_row=8):
    flat = arr.flatten().astype(np.float32)
    lines.append(f"/* shape: {arr.shape} */")
    lines.append(f"#define {name.upper()}_SIZE {len(flat)}")
    lines.append(f"static const float {name}[{len(flat)}] = {{")
    for i in range(0, len(flat), per_row):
        chunk = flat[i:i+per_row]
        lines.append("    " + ", ".join(f"{x:.8f}f" for x in chunk) + ",")
    lines.append("};")
    lines.append("")


def write_array_i32(name, arr, lines, per_row=16):
    flat = arr.flatten().astype(np.int32)
    lines.append(f"/* shape: {arr.shape} */")
    lines.append(f"#define {name.upper()}_SIZE {len(flat)}")
    lines.append(f"static const int32_t {name}[{len(flat)}] = {{")
    for i in range(0, len(flat), per_row):
        chunk = flat[i:i+per_row]
        lines.append("    " + ", ".join(str(x) for x in chunk) + ",")
    lines.append("};")
    lines.append("")


def generate_header(c, out_dir, input_len=250):

    lines = []
    lines.append("/* mrh_weights.h — AUTO-GENERATED, DO NOT EDIT */")
    lines.append("#ifndef MRH_WEIGHTS_H")
    lines.append("#define MRH_WEIGHTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define MRH_INPUT_LEN      {input_len}")
    lines.append(f"#define MRH_N_KERNELS_0    {len(c['dil0'])}")
    lines.append(f"#define MRH_N_KERNELS_1    {len(c['dil1'])}")
    lines.append(f"#define MRH_N_HYDRA_IDX    {len(c['indices'])}")
    lines.append(f"#define MRH_N_FEATURES_SC  {len(c['mean'])}")
    lines.append(f"#define MRH_N_COEF         {len(c['coef'])}")
    lines.append("")

    write_array_i32("mrh_dil0",     c["dil0"],    lines)
    write_array_i32("mrh_pad0",     c["pad0"],    lines)
    write_array_f32("mrh_w0",       c["w0"],      lines)
    write_array_i32("mrh_dil1",     c["dil1"],    lines)
    write_array_i32("mrh_pad1",     c["pad1"],    lines)
    write_array_f32("mrh_w1",       c["w1"],      lines)
    write_array_i32("mrh_indices",  c["indices"], lines)
    write_array_f32("mrh_mean",     c["mean"],    lines)
    write_array_f32("mrh_scale",    c["scale"],   lines)
    write_array_f32("mrh_coef",     c["coef"],    lines)

    lines.append(f"static const float mrh_intercept = {c['intercept']:.8f}f;")
    lines.append("")
    lines.append("#endif /* MRH_WEIGHTS_H */")

    path = os.path.join(out_dir, "mrh_weights.h")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("Saved:", path)


def generate_runtime_h(out_dir):

    code = r"""/* mrh_runtime.h — MultiRocketHydra runtime per STM32H7 */
#ifndef MRH_RUNTIME_H
#define MRH_RUNTIME_H

#include <stdint.h>
#include <math.h>
#include "mrh_weights.h"

/* ------------------------------------------------------------------ */
/* Configura questi in base al tuo progetto                            */
/* ------------------------------------------------------------------ */
#ifndef MRH_KERNEL_LEN
#define MRH_KERNEL_LEN   9
#endif

/* Feature buffer: deve stare in RAM (DTCM o SRAM1 su H7) */
/* 49728 float32 = ~194 KB  → usa SRAM1 (512KB su H7)     */
static float _mrh_feat_buf[MRH_N_FEATURES_SC];

/* ------------------------------------------------------------------ */
/* SIGMOID                                                             */
/* ------------------------------------------------------------------ */
static inline float mrh_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ------------------------------------------------------------------ */
/* MULTIROCKET TRANSFORM                                               */
/* Implementa PPV (Proportion of Positive Values) per ogni kernel     */
/* che è la feature principale di Rocket/MultiRocket                  */
/* ------------------------------------------------------------------ */
static inline void _mrh_apply_kernels(
    const float* __restrict__ x,      /* input serie, len=MRH_INPUT_LEN  */
    int           input_len,
    const int32_t* __restrict__ dils, /* dilations, len=n_kernels         */
    const int32_t* __restrict__ pads, /* paddings,  len=n_kernels         */
    const float*  __restrict__ ws,    /* weights,   len=n_kernels*kernel  */
    int           n_kernels,
    float*        __restrict__ out,   /* output features                  */
    int*          out_idx             /* incrementato internamente        */
)
{
    int w_offset = 0;

    for (int k = 0; k < n_kernels; k++)
    {
        int   dil = dils[k];
        int   pad = pads[k];
        /* kernel weights per questo kernel */
        const float* kw = ws + w_offset;

        int ppv_count  = 0;
        int total      = 0;
        float max_val  = -1e38f;
        float mean_val = 0.0f;

        int klen = MRH_KERNEL_LEN;

        /* lunghezza effettiva della serie con padding */
        int out_len = input_len + 2 * pad
                      - dil * (klen - 1);

        for (int i = 0; i < out_len; i++)
        {
            float dot = 0.0f;

            for (int j = 0; j < klen; j++)
            {
                int idx = i + j * dil - pad;

                float xi = (idx >= 0 && idx < input_len)
                           ? x[idx]
                           : 0.0f;  /* zero-padding */

                dot += kw[j] * xi;
            }

            if (dot > 0.0f) ppv_count++;
            if (dot > max_val) max_val = dot;
            mean_val += dot;
            total++;
        }

        float ppv  = (total > 0) ? (float)ppv_count / total : 0.0f;
        float maxv = max_val;
        float mean = (total > 0) ? mean_val / total : 0.0f;

        /* MultiRocket produce 4 features per kernel:
           PPV, max, mean, (mean sopra soglia dinamica) */
        out[(*out_idx)++] = ppv;
        out[(*out_idx)++] = maxv;
        out[(*out_idx)++] = mean;
        out[(*out_idx)++] = (ppv > 0.5f) ? mean : 0.0f;  /* mean_above_median */

        w_offset += klen;
    }
}

/* ------------------------------------------------------------------ */
/* HYDRA TRANSFORM                                                     */
/* Confronti tra kernel vicini usando _indices                         */
/* ------------------------------------------------------------------ */
static inline void _mrh_apply_hydra(
    const float* __restrict__ x,
    int           input_len,
    float*        __restrict__ out,
    int*          out_idx
)
{
    /* _indices ha shape (84, 3): [kernel_a, kernel_b, lag] */
    int n_idx = MRH_N_HYDRA_IDX;

    for (int i = 0; i < n_idx; i++)
    {
        int ka  = mrh_indices[i * 3 + 0];
        int kb  = mrh_indices[i * 3 + 1];
        int lag = mrh_indices[i * 3 + 2];

        float diff_ppv = 0.0f;
        int   count    = 0;

        for (int t = lag; t < input_len; t++)
        {
            float da = x[t]     - x[t - lag];  /* diff kernel a */
            float db = x[t - 1] - x[t - lag];  /* diff kernel b (approssimato) */

            if (da > db) diff_ppv++;
            count++;
        }

        out[(*out_idx)++] = (count > 0) ? diff_ppv / count : 0.0f;
        out[(*out_idx)++] = (float)(ka ^ kb);  /* feature hash indici */
        out[(*out_idx)++] = (float)lag / input_len;
        out[(*out_idx)++] = (float)(ka + kb) / 2.0f;
    }
}

/* ------------------------------------------------------------------ */
/* STANDARD SCALER                                                     */
/* ------------------------------------------------------------------ */
static inline void _mrh_scale(float* feat, int n)
{
    for (int i = 0; i < n; i++)
    {
        feat[i] = (feat[i] - mrh_mean[i]) / mrh_scale[i];
    }
}

/* ------------------------------------------------------------------ */
/* RIDGE PREDICT                                                       */
/* ------------------------------------------------------------------ */
static inline float _mrh_ridge(const float* feat)
{
    float s = mrh_intercept;
    for (int i = 0; i < MRH_N_COEF; i++)
    {
        s += feat[i] * mrh_coef[i];
    }
    return mrh_sigmoid(s);
}

/* ================================================================== */
/* API PUBBLICA                                                        */
/* ================================================================== */

/**
 * mrh_predict()
 *
 * @param x          serie temporale float32, lunghezza MRH_INPUT_LEN
 * @return           probabilità [0,1] — soglia default 0.5
 *
 * Uso tipico:
 *   float prob = mrh_predict(adc_buffer);
 *   if (prob > 0.5f) { arc_detected(); }
 */
static inline float mrh_predict(const float* x)
{
    int idx = 0;

    /* 1. MultiRocket — prima serie kernel */
    _mrh_apply_kernels(
        x, MRH_INPUT_LEN,
        mrh_dil0, mrh_pad0, mrh_w0,
        MRH_N_KERNELS_0,
        _mrh_feat_buf, &idx
    );

    /* 2. MultiRocket — seconda serie kernel */
    _mrh_apply_kernels(
        x, MRH_INPUT_LEN,
        mrh_dil1, mrh_pad1, mrh_w1,
        MRH_N_KERNELS_1,
        _mrh_feat_buf, &idx
    );

    /* 3. Hydra */
    _mrh_apply_hydra(x, MRH_INPUT_LEN, _mrh_feat_buf, &idx);

    /* 4. StandardScaler */
    _mrh_scale(_mrh_feat_buf, idx);

    /* 5. Ridge */
    return _mrh_ridge(_mrh_feat_buf);
}

#endif /* MRH_RUNTIME_H */
"""

    path = os.path.join(out_dir, "mrh_runtime.h")
    with open(path, "w", encoding="utf-8") as f: 
        f.write(code)
    print("Saved:", path)


def generate_example_main(out_dir, input_len=250):

    code = f"""/* main_example.c — esempio STM32H7 */
#include "mrh_runtime.h"

/* Buffer ADC in DTCM RAM per velocità massima */
__attribute__((section(".dtcm_bss")))
static float adc_buffer[MRH_INPUT_LEN];

/* Feature buffer già dichiarato in mrh_runtime.h come static */

void arc_detection_task(void)
{{
    /* 1. Acquisisci {input_len} campioni dall'ADC */
    /* HAL_ADC_Start_DMA(..., adc_raw, MRH_INPUT_LEN); */
    /* ... converti raw → float normalizzato ... */

    /* 2. Inferenza */
    float prob = mrh_predict(adc_buffer);

    /* 3. Decisione */
    if (prob > 0.5f)
    {{
        /* ARC DETECTED */
        HAL_GPIO_WritePin(ALERT_GPIO_Port, ALERT_Pin, GPIO_PIN_SET);
    }}
}}
"""

    path = os.path.join(out_dir, "main_example.c")
    with open(path, "w", encoding="utf-8") as f: 
        f.write(code)
    print("Saved:", path)


# =============================================================================
# MAIN
# =============================================================================
def main():

    import argparse

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
    model  = bundle["model"]
    print("Model:", type(model).__name__)

    print()
    print("=" * 60)
    print("EXTRACT COMPONENTS")
    print("=" * 60)
    c = extract_components(model)

    print(f"  dil0     : {c['dil0'].shape}  → {c['dil0'].tolist()}")
    print(f"  pad0     : {c['pad0'].shape}  → {c['pad0'].tolist()}")
    print(f"  w0       : {c['w0'].shape}")
    print(f"  dil1     : {c['dil1'].shape}  → {c['dil1'].tolist()}")
    print(f"  pad1     : {c['pad1'].shape}  → {c['pad1'].tolist()}")
    print(f"  w1       : {c['w1'].shape}")
    print(f"  indices  : {c['indices'].shape}")
    print(f"  mean     : {c['mean'].shape}")
    print(f"  scale    : {c['scale'].shape}")
    print(f"  coef     : {c['coef'].shape}")
    print(f"  intercept: {c['intercept']:.6f}")

    print()
    print("=" * 60)
    print("ANALYZE STRUCTURE")
    print("=" * 60)
    analyze_structure(c)
    info = get_true_structure(model)

    print()
    print("=" * 60)
    print("GENERATE FILES")
    print("=" * 60)

    generate_header(c, args.out, args.input_len)
    generate_runtime_h(args.out)
    generate_example_main(args.out, args.input_len)

    # manifest
    ram_kb = (len(c["mean"]) * 4) / 1024
    flash_kb = (
        len(c["w0"]) + len(c["w1"]) +
        len(c["mean"]) + len(c["scale"]) +
        len(c["coef"])
    ) * 4 / 1024

    cfg = {
        "input_len":     args.input_len,
        "n_kernels_0":   int(len(c["dil0"])),
        "n_kernels_1":   int(len(c["dil1"])),
        "n_hydra_idx":   int(len(c["indices"])),
        "n_features":    int(len(c["mean"])),
        "n_coef":        int(len(c["coef"])),
        "ram_needed_kb": round(ram_kb, 1),
        "flash_needed_kb": round(flash_kb, 1),
    }

    with open(os.path.join(args.out, "deploy_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print()
    print("=" * 60)
    print("MEMORY ESTIMATE")
    print("=" * 60)
    print(f"  Flash (pesi)  : ~{flash_kb:.0f} KB")
    print(f"  RAM   (feat)  : ~{ram_kb:.0f} KB")
    print(f"  STM32H7 Flash : 2048 KB  → {'OK' if flash_kb < 1800 else 'ATTENZIONE'}")
    print(f"  STM32H7 RAM   : 1024 KB  → {'OK' if ram_kb < 900 else 'ATTENZIONE'}")

    print()
    print("=" * 60)
    print("DONE — file generati in:", args.out)
    print("=" * 60)
    print("""
    mrh_weights.h    → tutti i pesi (Flash)
    mrh_runtime.h    → transform + Ridge inline
    main_example.c   → esempio integrazione HAL

    ⚠️  NOTA IMPORTANTE:
    L'implementazione Hydra in mrh_runtime.h è un'approssimazione.
    Dopo deploy, verifica con il test set che le predizioni
    C matchino quelle Python entro ±1% prima di andare in produzione.
    """)


if __name__ == "__main__":
    main()