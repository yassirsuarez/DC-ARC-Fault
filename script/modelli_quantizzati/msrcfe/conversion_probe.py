#!/usr/bin/env python3
"""
conversion_probe.py
===================
Strumenti per separare il costo di conversione FP32 <-> INT8 (nodi
QuantizeLinear / DequantizeLinear ai bordi del grafo) dal costo di calcolo
nei benchmark su STM32.

DOVE GIRA
    Sul PC, con Python >= 3.9 e `pip install onnx onnxruntime numpy`.
    Non serve la board, non servono i tool ST e non serve la GPU.
    Lo script PRODUCE o ANALIZZA file .onnx: la misura sulla board resta
    quella di ST Edge AI Developer Cloud (stessa scheda, stessa procedura
    usata per i benchmark gia' fatti).

SOTTOCOMANDI
    inspect    analizza il modello INT8 scaricato dal Developer Cloud
               (model_int8.onnx): trova i nodi di conversione ai bordi,
               stima il loro costo e, se dai i cicli misurati, calcola i
               cicli per MAC.
    strip-io   crea una variante con ingresso e uscita gia' INT8 (toglie il
               Quantize iniziale e il Dequantize finale) e verifica che
               l'output coincida con quello del modello originale.
               Benchmark(originale) - benchmark(variante) = costo di
               conversione misurato sulla board.
    relu-probe genera un modello minimo (solo Relu) + dataset di
               calibrazione, da far passare per la procedura normale
               (quantizzazione INT8 del Developer Cloud): il suo tempo e'
               praticamente solo conversione.

ESEMPI
    python conversion_probe.py inspect model_int8.onnx --cycles 41859280 --mhz 600
    python conversion_probe.py strip-io model_int8.onnx -o model_int8_ioint8.onnx
    python conversion_probe.py relu-probe --out probe --dataset arc_dataset_train.npz
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

# ─────────────────────────────────────────────
# Utilita' sul grafo
# ─────────────────────────────────────────────
Q_OPS = ("QuantizeLinear", "DequantizeLinear")
CONV_OPS = ("Conv", "QLinearConv", "ConvInteger")

NP_DTYPE = {
    TensorProto.INT8: np.int8,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT16: np.int16,
    TensorProto.UINT16: np.uint16,
}


def dtype_name(elem_type):
    return TensorProto.DataType.Name(elem_type)


def shape_of(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in dims]


def numel(shape):
    if any(not isinstance(d, int) or d <= 0 for d in shape):
        return None
    return int(np.prod(shape))


def initializers(model):
    """Dizionario nome -> ndarray per initializer e nodi Constant."""
    out = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and n.attribute:
            for a in n.attribute:
                if a.name == "value":
                    out[n.output[0]] = numpy_helper.to_array(a.t)
    return out


def build_maps(model):
    consumers, producers = defaultdict(list), {}
    for n in model.graph.node:
        for i in n.input:
            consumers[i].append(n)
        for o in n.output:
            producers[o] = n
    return consumers, producers


def real_inputs(model):
    init_names = {i.name for i in model.graph.initializer}
    return [i for i in model.graph.input if i.name not in init_names]


def find_boundaries(model):
    """Q ai bordi d'ingresso e DQ ai bordi d'uscita del grafo."""
    consumers, producers = build_maps(model)
    in_q = []
    for vi in real_inputs(model):
        cons = consumers[vi.name]
        if cons and all(c.op_type == "QuantizeLinear" and c.input[0] == vi.name for c in cons):
            in_q.append((vi, cons[0]))
    out_dq = []
    for vi in model.graph.output:
        p = producers.get(vi.name)
        if p is not None and p.op_type == "DequantizeLinear":
            out_dq.append((vi, p))
    return in_q, out_dq


def rename_tensor(graph, old, new):
    for n in graph.node:
        for k, name in enumerate(n.input):
            if name == old:
                n.input[k] = new
        for k, name in enumerate(n.output):
            if name == old:
                n.output[k] = new
    for vi in graph.value_info:
        if vi.name == old:
            vi.name = new


def zp_dtype(node, inits):
    """Tipo del tensore quantizzato, ricavato dallo zero point del nodo Q/DQ."""
    if len(node.input) > 2 and node.input[2] in inits:
        return {np.dtype(v): k for k, v in NP_DTYPE.items()}.get(inits[node.input[2]].dtype, TensorProto.INT8)
    return TensorProto.UINT8  # default ONNX se lo zero point manca


# ─────────────────────────────────────────────
# inspect
# ─────────────────────────────────────────────
def conv_macs(model, inits, producers):
    """MAC totali dei nodi Conv (richiede shape inferite); None se non calcolabili."""
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception:
        return None
    vi_map = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    total = 0
    for n in model.graph.node:
        if n.op_type not in CONV_OPS:
            continue
        w_name = n.input[3] if n.op_type == "QLinearConv" else n.input[1]
        w = inits.get(w_name)
        if w is None and w_name in producers and producers[w_name].op_type == "DequantizeLinear":
            w = inits.get(producers[w_name].input[0])
        out = vi_map.get(n.output[0])
        if w is None or out is None:
            return None
        out_shape = shape_of(out)
        if numel(out_shape) is None:
            return None
        # Conv: W = (O, I/g, k...) ; MAC = N * O * L * (I/g) * prod(k)
        total += numel(out_shape) * int(np.prod(w.shape[1:]))
    return total


def cmd_inspect(args):
    model = onnx.load(args.model)
    inits = initializers(model)
    consumers, producers = build_maps(model)
    in_q, out_dq = find_boundaries(model)

    print(f"Modello: {args.model}  (opset {[o.version for o in model.opset_import]}, IR {model.ir_version})")
    for vi in real_inputs(model):
        print(f"  Ingresso : {vi.name:<20} {dtype_name(vi.type.tensor_type.elem_type):<8} {shape_of(vi)}")
    for vi in model.graph.output:
        print(f"  Uscita   : {vi.name:<20} {dtype_name(vi.type.tensor_type.elem_type):<8} {shape_of(vi)}")

    ops = Counter(n.op_type for n in model.graph.node)
    print("\nOperatori nel grafo:")
    for op, c in ops.most_common():
        print(f"  {op:<22} {c}")

    n_q = sum(1 for n in model.graph.node if n.op_type == "QuantizeLinear")
    n_dq = sum(1 for n in model.graph.node if n.op_type == "DequantizeLinear")
    print(f"\nNodi di conversione: {n_q} QuantizeLinear, {n_dq} DequantizeLinear")
    print(f"  ai bordi d'ingresso: {len(in_q)} Q   |   ai bordi d'uscita: {len(out_dq)} DQ")
    if not in_q and not out_dq:
        print("  Nessun nodo Q/DQ al bordo: ingresso/uscita sono gia' quantizzati,")
        print("  oppure il formato del grafo non e' QDQ/QOperator.")

    # Stima analitica del costo di conversione ai bordi
    n_in = sum(numel(shape_of(vi)) or 0 for vi, _ in in_q)
    n_out = sum(numel(shape_of(vi)) or 0 for vi, _ in out_dq)
    cyc = n_in * args.cycles_q + n_out * args.cycles_dq
    print("\nStima analitica del costo di conversione ai bordi (NON e' una misura):")
    print(f"  elementi quantizzati in ingresso : {n_in}")
    print(f"  elementi dequantizzati in uscita : {n_out}")
    print(f"  ipotesi: {args.cycles_q:g} cicli/elem (Q), {args.cycles_dq:g} cicli/elem (DQ)")
    print(f"  cicli stimati                    : {cyc:,.0f}  ->  {cyc / args.mhz:,.1f} us a {args.mhz:g} MHz")
    if args.total_ms:
        print(f"  incidenza sul tempo totale       : {100 * (cyc / args.mhz / 1000) / args.total_ms:.3f}% di {args.total_ms} ms")

    macs = conv_macs(model, inits, producers)
    if macs:
        print(f"\nMAC delle convoluzioni: {macs:,}")
        if args.cycles:
            print(f"  cicli misurati: {args.cycles:,}  ->  {args.cycles / macs:.1f} cicli/MAC")
            print("  (un valore molto alto indica che il tempo non e' dominato dalle convoluzioni)")
    elif args.cycles:
        print("\nMAC non calcolabili (shape non inferibili): cicli/MAC non disponibile.")


# ─────────────────────────────────────────────
# strip-io
# ─────────────────────────────────────────────
def quant_params(node, inits):
    """(scale, zero_point) per-tensor di un nodo Q/DQ; None se per-asse."""
    scale = inits.get(node.input[1])
    zp = inits.get(node.input[2]) if len(node.input) > 2 else np.array(0)
    if scale is None or zp is None or scale.size != 1 or zp.size != 1:
        return None
    return float(scale.reshape(-1)[0]), int(zp.reshape(-1)[0])


def cmd_strip_io(args):
    import onnxruntime as ort

    orig = onnx.load(args.model)
    model = onnx.load(args.model)
    inits = initializers(model)
    in_q, out_dq = find_boundaries(model)
    if not in_q and not out_dq:
        sys.exit("Nessun nodo Q/DQ ai bordi: niente da rimuovere.")

    g = model.graph
    in_params, out_params = {}, {}

    # Ingressi: x(float) -> Q -> ...   diventa   x(int8) -> ...
    for vi, q in in_q:
        in_params[vi.name] = quant_params(q, inits)
        q_out, elem = q.output[0], zp_dtype(q, inits)
        shape = vi.type.tensor_type.shape
        g.node.remove(q)
        rename_tensor(g, q_out, vi.name)
        vi.type.tensor_type.elem_type = elem
        vi.type.tensor_type.shape.CopyFrom(shape)

    # Uscite: ... -> DQ -> y(float)   diventa   ... -> y(int8)
    for vi, dq in out_dq:
        out_params[vi.name] = quant_params(dq, inits)
        dq_in, elem = dq.input[0], zp_dtype(dq, inits)
        g.node.remove(dq)
        rename_tensor(g, dq_in, vi.name)
        vi.type.tensor_type.elem_type = elem

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"Salvato: {args.output}")
    for vi, _ in in_q:
        print(f"  ingresso '{vi.name}' ora {dtype_name(vi.type.tensor_type.elem_type)}  (scale, zp) = {in_params[vi.name]}")
    for vi, _ in out_dq:
        print(f"  uscita   '{vi.name}' ora {dtype_name(vi.type.tensor_type.elem_type)}  (scale, zp) = {out_params[vi.name]}")

    # Verifica numerica: stessa uscita (dopo dequantizzazione) del modello originale
    if any(p is None for p in list(in_params.values()) + list(out_params.values())):
        print("\nVerifica numerica saltata: parametri di quantizzazione per-asse ai bordi.")
        return
    rng = np.random.default_rng(0)
    s0 = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    feed0, feed1 = {}, {}
    for vi in real_inputs(orig):
        shp = [d if isinstance(d, int) and d > 0 else 1 for d in shape_of(vi)]
        x = rng.normal(1.0, 0.05, size=shp).astype(np.float32)
        feed0[vi.name] = x
        scale, zp = in_params[vi.name]
        info = np.iinfo(NP_DTYPE[zp_dtype(next(q for v, q in in_q if v.name == vi.name), inits)])
        feed1[vi.name] = np.clip(np.round(x / scale) + zp, info.min, info.max).astype(
            NP_DTYPE[zp_dtype(next(q for v, q in in_q if v.name == vi.name), inits)]
        )
    y0 = s0.run(None, feed0)
    y1 = s1.run(None, feed1)
    worst = 0.0
    for vi, yq in zip(model.graph.output, y1):
        scale, zp = out_params[vi.name]
        y1f = (yq.astype(np.float32) - zp) * scale
        ref = y0[[o.name for o in orig.graph.output].index(vi.name)]
        worst = max(worst, float(np.abs(ref - y1f).max()))
    print(f"\nVerifica numerica: differenza massima originale vs variante = {worst:.3e}")
    print("  (attesa ~0: i due modelli eseguono lo stesso calcolo interno)")
    print("\nPassi successivi:")
    print("  1. Carica ENTRAMBI i modelli (originale e variante) su ST Edge AI Developer Cloud,")
    print("     scegliendo la stessa scheda, e lancia il benchmark.")
    print("  2. Costo di conversione = cicli(originale) - cicli(variante).")
    print("  3. Se il tool rifiuta la variante o la ricompila diversamente, il confronto non e'")
    print("     valido: usa 'relu-probe' o il contatore di cicli DWT nel firmware.")


# ─────────────────────────────────────────────
# relu-probe
# ─────────────────────────────────────────────
def cmd_relu_probe(args):
    os.makedirs(args.out, exist_ok=True)
    L = args.length
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, L])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, L])
    graph = helper.make_graph([helper.make_node("Relu", ["input"], ["output"])], "conversion_probe_relu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7  # come l'export PyTorch opset 13 usato per MS-RCFE
    onnx.checker.check_model(model)
    onnx_path = os.path.join(args.out, "probe_relu_fp32.onnx")
    onnx.save(model, onnx_path)

    rng = np.random.default_rng(42)
    if args.dataset:
        X = np.load(args.dataset)["X"].astype(np.float32)
        X = X[rng.choice(len(X), min(args.n_cal, len(X)), replace=False)][:, :L]
    else:
        X = rng.normal(1.0, 0.05, size=(args.n_cal, L)).astype(np.float32)
    cal_path = os.path.join(args.out, "calibration_probe.npz")
    np.savez(cal_path, input=X[:, np.newaxis, np.newaxis, :])  # (N,1,1,L), come calibration_msrcfe.npz

    print(f"Creati:\n  {onnx_path}\n  {cal_path}  shape {(len(X), 1, 1, L)}")
    print("\nCome usarli:")
    print("  1. Developer Cloud: carica probe_relu_fp32.onnx, lancia il benchmark FP32 sulla scheda scelta.")
    print("  2. Stessa scheda: quantizza INT8 per-channel con calibration_probe.npz e lancia il benchmark.")
    print("  3. Il calcolo e' un solo Relu su 1000 campioni (trascurabile): la differenza di cicli")
    print("     tra INT8 e FP32 e' essenzialmente il costo di Quantize + Dequantize ai bordi.")
    print("  Nota: il tool potrebbe fondere o rimuovere nodi; controlla nel report cosa e' stato compilato.")


# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Separa costo di conversione e calcolo nei benchmark ONNX/STM32.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inspect", help="analizza un modello INT8 (QDQ/QOperator)")
    a.add_argument("model")
    a.add_argument("--cycles", type=int, help="cicli misurati sulla board (colonna 'cycles' del report)")
    a.add_argument("--mhz", type=float, default=600.0, help="clock della CPU [MHz] (H7S78: 600, N6: 800)")
    a.add_argument("--total-ms", type=float, help="tempo totale misurato [ms], per l'incidenza percentuale")
    a.add_argument("--cycles-q", type=float, default=10.0, help="ipotesi cicli/elemento per Quantize")
    a.add_argument("--cycles-dq", type=float, default=6.0, help="ipotesi cicli/elemento per Dequantize")
    a.set_defaults(func=cmd_inspect)

    b = sub.add_parser("strip-io", help="variante con ingresso/uscita INT8")
    b.add_argument("model")
    b.add_argument("-o", "--output", required=True)
    b.set_defaults(func=cmd_strip_io)

    c = sub.add_parser("relu-probe", help="modello minimo per misurare la sola conversione")
    c.add_argument("--out", default="probe")
    c.add_argument("--length", type=int, default=1000, help="campioni per finestra")
    c.add_argument("--n-cal", type=int, default=200, help="finestre di calibrazione")
    c.add_argument("--dataset", help="arc_dataset_train.npz (chiave X) per calibrare su dati reali")
    c.set_defaults(func=cmd_relu_probe)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
