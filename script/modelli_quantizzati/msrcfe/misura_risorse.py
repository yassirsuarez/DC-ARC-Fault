#!/usr/bin/env python3
"""
misura_risorse.py
=======================
Benchmark latenza e stima risorse embedded per pipeline CNN (ONNX) + Ridge.

Misura:
    - Latenza inferenza CNN (ONNX): mean, std, min, max, p95
    - Latenza Ridge (simulato C):   mean, std, min, max, p95
    - Latenza pipeline totale:      mean, std, min, max, p95
    - Delta RAM durante inferenza
    - Stima risorse embedded STM32 (Flash + RAM)

Output:
    benchmark_latency.csv     latenze complete con percentili
    benchmark_resources.csv   stima risorse embedded STM32
    benchmark_report.txt      report testuale leggibile

USO:
    python misura_risorse.py \
        --model  path/to/model.onnx \
        --coef   path/to/ridge_coef.npy \
        --bias   path/to/ridge_intercept.npy \
        --out    ./stm32
"""

import argparse
import csv
import os
import time
import numpy as np
import psutil
import onnxruntime as ort

# Vincolo temporale reale imposto dalla pipeline di classificazione real-time
# (vedi build_dataset_new.py: WINDOW_S = 0.10 s, la finestra di classificazione).
# Lo standard UL 1699B richiede che il dispositivo rilevi l'arco entro il tempo
# limite previsto dai suoi criteri di sicurezza; nella nostra pipeline questo si
# traduce nel vincolo operativo di elaborare una finestra di 100 ms prima che
# arrivi la successiva. NON è un requisito a 50 Hz: quella soglia (20 ms) era
# un valore arbitrario, pensato per il benchmark locale su CPU host (dove la
# pipeline gira in meno di 1 ms) e non ha alcun riscontro nei parametri reali
# del progetto o nello standard citato.
REALTIME_WINDOW_MS = 100.0  # WINDOW_S * 1000, da build_dataset_new.py


# =============================================================================
# TIMING
# =============================================================================

def measure_latency(fn, warmup=10, runs=100):
    """
    Esegue warmup silenzioso, poi misura runs esecuzioni.
    Restituisce un dict con mean, std, min, max, p50, p95 in ms.
    """
    for _ in range(warmup):
        fn()

    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1000.0  # → ms

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
        "p50_ms":  float(np.percentile(times, 50)),
        "p95_ms":  float(np.percentile(times, 95)),
    }


# =============================================================================
# RAM
# =============================================================================

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def measure_ram_delta(fn, samples=10):
    """Misura il delta RAM medio su più campionamenti (riduce rumore del GC)."""
    deltas = []
    for _ in range(samples):
        before = get_ram_mb()
        fn()
        after  = get_ram_mb()
        deltas.append(after - before)
    return float(np.mean(deltas))


# =============================================================================
# STIMA RISORSE EMBEDDED STM32
# =============================================================================

def estimate_stm32_resources(sess, ridge_weights, ridge_bias, input_shape):
    """
    Stima Flash e RAM necessarie per eseguire la pipeline su STM32.

    Flash:
        - Pesi ONNX (dal file su disco)
        - Pesi Ridge: coef + intercept (float32)

    RAM:
        - Tensor di input (float32)
        - Tensor di output CNN (feature map, float32)
        - Buffer Ridge output (1 float)
        - Stack attivazioni interne (stima 2x output CNN)

    Nota: sono stime conservative, non tengono conto di quantizzazione
    o ottimizzazioni ST Edge AI.
    """
    # ── Flash ─────────────────────────────────────────────────────────────────
    onnx_path   = sess._model_path if hasattr(sess, "_model_path") else None
    onnx_flash_kb = os.path.getsize(onnx_path) / 1024 if onnx_path and os.path.exists(onnx_path) else 0.0

    n_ridge_params  = ridge_weights.size + ridge_bias.size
    ridge_flash_kb  = n_ridge_params * 4 / 1024   # float32 = 4 bytes

    total_flash_kb  = onnx_flash_kb + ridge_flash_kb

    # ── RAM ───────────────────────────────────────────────────────────────────
    input_elements  = int(np.prod(input_shape))
    input_ram_kb    = input_elements * 4 / 1024

    # Output CNN: dimensione del primo output del modello
    output_meta     = sess.get_outputs()[0]
    output_shape    = [d if isinstance(d, int) and d > 0 else 1
                       for d in output_meta.shape]
    output_elements = int(np.prod(output_shape))
    output_ram_kb   = output_elements * 4 / 1024

    # Buffer attivazioni interne (stima: 2x output, conservativa)
    internal_ram_kb = output_ram_kb * 2

    total_ram_kb    = input_ram_kb + output_ram_kb + internal_ram_kb

    return {
        "onnx_flash_kb":     round(onnx_flash_kb,    1),
        "ridge_flash_kb":    round(ridge_flash_kb,   1),
        "total_flash_kb":    round(total_flash_kb,   1),
        "total_flash_mb":    round(total_flash_kb / 1024, 3),
        "input_ram_kb":      round(input_ram_kb,     2),
        "cnn_output_ram_kb": round(output_ram_kb,    2),
        "internal_ram_kb":   round(internal_ram_kb,  2),
        "total_ram_kb":      round(total_ram_kb,     2),
        "n_ridge_params":    n_ridge_params,
        "stm32_flash_ok":    total_flash_kb < 2048,     # <2 MB → Flash interna H7
        "stm32_ram_ok":      total_ram_kb   < 512,      # <512 KB → RAM tipica H7
    }


# =============================================================================
# BENCHMARK PRINCIPALE
# =============================================================================

def run_benchmark(sess, ridge_weights, ridge_bias, x_sample, warmup, runs):

    x = x_sample.astype(np.float32)

    # Estrai feature CNN una volta (usate dal ridge standalone)
    feat = sess.run(None, {"input": x})[0]

    # ── CNN ───────────────────────────────────────────────────────────────────
    def cnn_forward():
        sess.run(None, {"input": x})

    print(f"  Misuro CNN       ({runs} run, warmup={warmup})...")
    cnn_stats = measure_latency(cnn_forward, warmup=warmup, runs=runs)

    # ── Ridge ─────────────────────────────────────────────────────────────────
    def ridge_forward():
        score = np.dot(feat, ridge_weights) + ridge_bias
        return 1 if score > 0 else 0

    print(f"  Misuro Ridge     ({runs} run, warmup={warmup})...")
    ridge_stats = measure_latency(ridge_forward, warmup=warmup, runs=runs)

    # ── Pipeline totale ───────────────────────────────────────────────────────
    def full_pipeline():
        f     = sess.run(None, {"input": x})[0]
        score = np.dot(f, ridge_weights) + ridge_bias
        return 1 if score > 0 else 0

    print(f"  Misuro pipeline  ({runs} run, warmup={warmup})...")
    total_stats = measure_latency(full_pipeline, warmup=warmup, runs=runs)

    # ── RAM delta ─────────────────────────────────────────────────────────────
    print("  Misuro RAM delta...")
    ram_delta_mb = measure_ram_delta(full_pipeline, samples=20)

    return {
        "cnn":      cnn_stats,
        "ridge":    ridge_stats,
        "pipeline": total_stats,
        "ram_delta_mb": ram_delta_mb,
    }


# =============================================================================
# EXPORT CSV
# =============================================================================

def export_latency_csv(latency, out_path):
    fields = ["stage", "mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"]
    rows = [
        {"stage": "CNN (ONNX)",       **latency["cnn"]},
        {"stage": "Ridge (C sim)",     **latency["ridge"]},
        {"stage": "Pipeline totale",   **latency["pipeline"]},
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    print(f"  CSV latenze    : {out_path}")


def export_resources_csv(resources, out_path):
    fields = list(resources.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(resources)
    print(f"  CSV risorse    : {out_path}")


# =============================================================================
# EXPORT TXT
# =============================================================================

def export_txt_report(latency, resources, ram_delta_mb, args, out_path):
    sep  = "=" * 64
    thin = "-" * 64

    def stat_row(label, stats):
        return (f"  {label:<18} "
                f"mean={stats['mean_ms']:7.3f}ms  "
                f"std={stats['std_ms']:6.3f}  "
                f"min={stats['min_ms']:6.3f}  "
                f"max={stats['max_ms']:6.3f}  "
                f"p95={stats['p95_ms']:6.3f}")

    lines = [
        sep,
        "  BENCHMARK CNN + RIDGE — REPORT",
        sep, "",
        f"  Modello ONNX : {args.model}",
        f"  Input shape  : {args.input_shape}",
        f"  Warmup runs  : {args.warmup}",
        f"  Bench runs   : {args.runs}",
        "", thin,
        "  TABELLA 1 — LATENZE (ms)",
        thin,
        f"  {'Stage':<18} {'mean':>10}  {'std':>8}  {'min':>8}  {'max':>8}  {'p95':>8}",
        thin,
        stat_row("CNN (ONNX)",     latency["cnn"]),
        stat_row("Ridge (C sim)",  latency["ridge"]),
        stat_row("Pipeline tot.",  latency["pipeline"]),
        thin,
        f"  RAM delta inferenza : {ram_delta_mb:.3f} MB",
        "", thin,
        "  TABELLA 2 — STIMA RISORSE EMBEDDED STM32",
        thin,
        "  Flash",
        f"    ONNX model          : {resources['onnx_flash_kb']:>8.1f} KB",
        f"    Ridge pesi          : {resources['ridge_flash_kb']:>8.1f} KB",
        f"    Totale Flash        : {resources['total_flash_kb']:>8.1f} KB"
        f"  ({resources['total_flash_mb']:.3f} MB)",
        "",
        "  RAM (runtime)",
        f"    Input tensor        : {resources['input_ram_kb']:>8.2f} KB",
        f"    Output CNN          : {resources['cnn_output_ram_kb']:>8.2f} KB",
        f"    Buffer interno est. : {resources['internal_ram_kb']:>8.2f} KB",
        f"    Totale RAM          : {resources['total_ram_kb']:>8.2f} KB",
        "",
        "  Compatibilità STM32H7 (2 MB Flash int. / 512 KB RAM)",
        f"    Flash OK            : {'SI' if resources['stm32_flash_ok'] else 'NO — supera 2 MB Flash interna'}",
        f"    RAM OK              : {'SI' if resources['stm32_ram_ok']   else 'NO — supera 512 KB RAM'}",
        "", thin,
        "  TABELLA 3 — THROUGHPUT STIMATO",
        thin,
    ]

    # Throughput (inferenze/secondo) basato sulla pipeline totale
    mean_ms  = latency["pipeline"]["mean_ms"]
    p95_ms   = latency["pipeline"]["p95_ms"]
    thr_mean = 1000.0 / mean_ms if mean_ms > 0 else 0
    thr_p95  = 1000.0 / p95_ms  if p95_ms  > 0 else 0
    lines += [
        f"  Throughput (mean) : {thr_mean:>8.1f} inf/s",
        f"  Throughput (p95)  : {thr_p95:>8.1f} inf/s",
        f"  Latenza budget    : {mean_ms:.3f} ms/inf  (vincolo: "
        f"{REALTIME_WINDOW_MS:.0f} ms/finestra, UL 1699B)  →  "
        f"{'adeguato' if mean_ms < REALTIME_WINDOW_MS else 'troppo lento'}",
        "",
        sep,
        "  Fine report",
        sep,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report TXT     : {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark latenza CNN (ONNX) + Ridge con stima risorse STM32"
    )
    parser.add_argument("--model",  required=True,
                        help="Path al file .onnx")
    parser.add_argument("--coef",   required=True,
                        help="Path a ridge_coef.npy")
    parser.add_argument("--bias",   required=True,
                        help="Path a ridge_intercept.npy")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1, 1, 1000],
                        help="Shape input (es. 1 1 1000)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Numero di run di warmup (default: 10)")
    parser.add_argument("--runs",   type=int, default=200,
                        help="Numero di run di misura (default: 200)")
    parser.add_argument("--out",    default=".",
                        help="Directory output (default: directory corrente)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Carica modello e pesi ────────────────────────────────────────────────
    print("\n── CARICAMENTO ──")
    print(f"  ONNX  : {args.model}")
    sess = ort.InferenceSession(args.model)
    # Salva path per la stima Flash
    sess._model_path = args.model

    print(f"  Coef  : {args.coef}")
    ridge_weights = np.load(args.coef).astype(np.float32).flatten()

    print(f"  Bias  : {args.bias}")
    ridge_bias = np.load(args.bias).astype(np.float32).flatten()

    x_sample = np.random.randn(*args.input_shape).astype(np.float32)
    print(f"  Input shape : {x_sample.shape}")

    # ── Benchmark latenze ────────────────────────────────────────────────────
    print("\n── BENCHMARK LATENZE ──")
    latency = run_benchmark(
        sess, ridge_weights, ridge_bias, x_sample,
        warmup=args.warmup, runs=args.runs
    )

    # ── Stima risorse embedded ───────────────────────────────────────────────
    print("\n── STIMA RISORSE STM32 ──")
    resources = estimate_stm32_resources(
        sess, ridge_weights, ridge_bias, args.input_shape)

    # ── Stampa a terminale ───────────────────────────────────────────────────
    print("\n── RISULTATI ──")
    stages = [
        ("CNN (ONNX)",    latency["cnn"]),
        ("Ridge (C sim)", latency["ridge"]),
        ("Pipeline tot.", latency["pipeline"]),
    ]
    hdr = f"  {'Stage':<18} {'mean':>8} {'std':>7} {'min':>7} {'max':>7} {'p95':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, s in stages:
        print(f"  {label:<18} "
              f"{s['mean_ms']:7.3f}  "
              f"{s['std_ms']:6.3f}  "
              f"{s['min_ms']:6.3f}  "
              f"{s['max_ms']:6.3f}  "
              f"{s['p95_ms']:6.3f}  ms")
    print(f"\n  RAM delta : {latency['ram_delta_mb']:.3f} MB")

    print("\n── RISORSE STM32 ──")
    print(f"  Flash totale : {resources['total_flash_kb']:.1f} KB  "
          f"({'OK' if resources['stm32_flash_ok'] else 'ATTENZIONE: >2MB'})")
    print(f"  RAM totale   : {resources['total_ram_kb']:.2f} KB  "
          f"({'OK' if resources['stm32_ram_ok'] else 'ATTENZIONE: >512KB'})")

    # ── Export ───────────────────────────────────────────────────────────────
    print("\n── EXPORT ──")
    export_latency_csv(
        latency,
        os.path.join(args.out, "benchmark_latency.csv")
    )
    export_resources_csv(
        resources,
        os.path.join(args.out, "benchmark_resources.csv")
    )
    export_txt_report(
        latency, resources, latency["ram_delta_mb"], args,
        os.path.join(args.out, "benchmark_report.txt")
    )

    print(f"\n  Output in: {args.out}/")
    print("     benchmark_latency.csv")
    print("     benchmark_resources.csv")
    print("     benchmark_report.txt")


if __name__ == "__main__":
    main()