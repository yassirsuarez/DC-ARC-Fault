/* rocket_transform.h - AUTO-GENERATED, DO NOT EDIT */
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
/* Feature buffer RAM — 12432 x 4 = 48.6 KB                    */
/* Metti in DTCM (.dtcm_bss) per velocita' massima su H7              */
/* ------------------------------------------------------------------ */
static float _rocket_feat[ROCKET_TOTAL_FEATURES];

/* ------------------------------------------------------------------ */
/* DIFFERENZA PRIMA ORDINE (np.diff equivalente)                       */
/* ------------------------------------------------------------------ */
static float _rocket_diff[249];

static inline void _rocket_compute_diff(const float* x, int n)
{
    for (int i = 0; i < n - 1; i++)
        _rocket_diff[i] = x[i + 1] - x[i];
}

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
{
    int bias_offset = 0;

    for (int d_idx = 0; d_idx < n_dilations; d_idx++)
    {
        int dilation = dilations[d_idx];
        int n_feat   = n_fpd[d_idx];
        int padding  = ((9 - 1) * dilation) / 2;
        int out_len  = input_len + 2 * padding - dilation * (9 - 1);

        /* Pre-calcola tutti i dot product per questo dilation */
        /* Buffer temporaneo sullo stack — max out_len ~250    */
        float dots[350];

        for (int k = 0; k < ROCKET_N_KERNELS; k++)
        {
            int i0 = rocket_indices[k][0];
            int i1 = rocket_indices[k][1];
            int i2 = rocket_indices[k][2];

            /* Calcola dot products per tutti i timestep */
            /* Loop CORRETTO: feature -> kernel */
            for (int f = 0; f < n_feat; f++)
            {
                for (int k = 0; k < ROCKET_N_KERNELS; k++)
                {
                    int i0 = rocket_indices[k][0];
                    int i1 = rocket_indices[k][1];
                    int i2 = rocket_indices[k][2];

                    float ppv = 0.0f;
                    for (int t = 0; t < out_len; t++)
                    {
                        int p0 = t + i0 * dilation - padding;
                        int p1 = t + i1 * dilation - padding;
                        int p2 = t + i2 * dilation - padding;

                        float v0 = (p0 >= 0 && p0 < input_len) ? x[p0] : 0.0f;
                        float v1 = (p1 >= 0 && p1 < input_len) ? x[p1] : 0.0f;
                        float v2 = (p2 >= 0 && p2 < input_len) ? x[p2] : 0.0f;

                        float dot = -v0 + 2.0f * v1 - v2;
                        if (dot > biases[bias_offset]) ppv += 1.0f;
                    }

                    out[(*out_idx)++] = (out_len > 0) ? ppv / out_len : 0.0f;
                    bias_offset++;
    }
}
        }
    }
}

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
{
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
}

#endif /* ROCKET_TRANSFORM_H */
