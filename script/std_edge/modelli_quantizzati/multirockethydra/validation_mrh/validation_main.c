/* validation_main.c
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
        printf("sample %2d: label=%d py=%.4f c=%.4f diff=%.6f %s\n",
               i, val_labels[i], py_prob, c_prob, diff,
               ok ? "OK" : "WARN");
    }

    printf("\nResult: %d/%d pass, max_diff=%.6f\n",
           pass, N_VALIDATION_SAMPLES, max_diff);

    if (pass == N_VALIDATION_SAMPLES)
        printf("VALIDATION PASSED\n");
    else
        printf("VALIDATION FAILED — controlla mrh_runtime.h\n");
}
