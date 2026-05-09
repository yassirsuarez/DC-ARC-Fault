/* main_example.c — esempio STM32H7 */
#include "mrh_runtime.h"

/* Buffer ADC in DTCM RAM per velocità massima */
__attribute__((section(".dtcm_bss")))
static float adc_buffer[MRH_INPUT_LEN];

/* Feature buffer già dichiarato in mrh_runtime.h come static */

void arc_detection_task(void)
{
    /* 1. Acquisisci 250 campioni dall'ADC */
    /* HAL_ADC_Start_DMA(..., adc_raw, MRH_INPUT_LEN); */
    /* ... converti raw → float normalizzato ... */

    /* 2. Inferenza */
    float prob = mrh_predict(adc_buffer);

    /* 3. Decisione */
    if (prob > 0.5f)
    {
        /* ARC DETECTED */
        HAL_GPIO_WritePin(ALERT_GPIO_Port, ALERT_Pin, GPIO_PIN_SET);
    }
}
