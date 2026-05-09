/* mrh_runtime.h — MultiRocketHydra runtime per STM32H7 */
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
