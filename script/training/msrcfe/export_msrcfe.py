#!/usr/bin/env python3
"""
export_msrcfe.py
====================
Export SOLO modello MSRCFE per STM32H7
✔ Fix dim=2 (no Squeeze bug)
✔ Opset 13 (ST Edge AI compatibile)  
✔ Verifica numerica PyTorch vs ONNX
✔ Shape statica per STM32 (batch=1)

esempio uso:
python export_msrcfe.py msrcfe_bundle.pkl arc_dataset_train.npz --out export_msrcfe
"""

import os
import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MSRCFE ARCH
# ─────────────────────────────────────────────
class MSRCFEFeatureExtractor(nn.Module):
    def __init__(self, n_kernels=32, kernel_sizes=[3, 5, 9], dilations=[1, 2, 4]):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            for d in dilations:
                self.convs.append(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=n_kernels,
                        kernel_size=k,
                        dilation=d,
                        padding=(k // 2) * d
                    )
                )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self._extract(x)

    def forward_no_unsqueeze(self, x):
        return self._extract(x)

    def _extract(self, x):
        feats = []
        for conv in self.convs:
            y = torch.relu(conv(x))          # (B, n_kernels, T)
            feats.append(y.max(dim=2).values) # (B, n_kernels) — FIX: dim=2
            feats.append(y.mean(dim=2))       # (B, n_kernels) — FIX: dim=2
        return torch.cat(feats, dim=1)        # (B, n_features)


# ─────────────────────────────────────────────
class MSRCFEONNXWrapper(torch.nn.Module):
    def __init__(self, msrcfe):
        super().__init__()
        self.msrcfe = msrcfe

    def forward(self, x):
        return self.msrcfe.forward_no_unsqueeze(x)


# ─────────────────────────────────────────────
def export_msrcfe_onnx(msrcfe, n_tp, out_dir):
    import onnx
    import onnxruntime as ort

    msrcfe.eval()
    wrapper = MSRCFEONNXWrapper(msrcfe)
    wrapper.eval()

    dummy = torch.zeros(1, 1, n_tp)
    path  = os.path.join(out_dir, "msrcfe.onnx")

    # ── EXPORT con shape statica (batch=1 fisso per STM32) ──
    torch.onnx.export(
        wrapper,
        dummy,
        path,
        opset_version=13,
        input_names=["input"],
        output_names=["features"],
        # Niente dynamic_axes → shape completamente statica
        # ST Edge AI preferisce shape fisse per la quantizzazione
        do_constant_folding=True,
        export_params=True,
    )

    # ── VERIFICA STRUTTURA GRAFO ──
    model_onnx = onnx.load(path)
    onnx.checker.check_model(model_onnx)
    log.info(f"✔ ONNX checker: OK")

    # Controlla che non ci siano nodi Squeeze problematici
    squeeze_nodes = [n for n in model_onnx.graph.node if n.op_type == "Squeeze"]
    if squeeze_nodes:
        log.warning(f"⚠ Trovati {len(squeeze_nodes)} nodi Squeeze — potrebbero causare problemi")
        for node in squeeze_nodes:
            log.warning(f"  Squeeze: inputs={list(node.input)}")
    else:
        log.info("✔ Nessun nodo Squeeze — grafo pulito")

    # ── VERIFICA NUMERICA PyTorch vs ONNX ──
    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()

    sess    = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    rel_diff = max_diff / (np.abs(pt_out).max() + 1e-8)

    log.info(f"✔ Verifica numerica:")
    log.info(f"  Max diff assoluta:  {max_diff:.2e}")
    log.info(f"  Max diff relativa:  {rel_diff:.2e}")

    if max_diff > 1e-4:
        log.warning("⚠ Differenza numerica alta — controlla l'architettura")
    else:
        log.info("✔ Output PyTorch ≈ Output ONNX — export corretto")

    # ── INFO GRAFO ──
    log.info(f"\n── Info ONNX ──")
    log.info(f"  Input:   {sess.get_inputs()[0].name}  {sess.get_inputs()[0].shape}")
    log.info(f"  Output:  {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")
    log.info(f"✔ MSRCFE ONNX salvato: {path}")

    return sess.get_outputs()[0].shape[-1]  # restituisce n_features


# ─────────────────────────────────────────────
def export_ridge_header(clf, out_dir):
    coef      = clf.coef_.flatten().astype(np.float32)
    intercept = float(clf.intercept_[0])
    path      = os.path.join(out_dir, "ridge_weights.h")

    with open(path, "w") as f:
        f.write("#ifndef RIDGE_WEIGHTS_H\n#define RIDGE_WEIGHTS_H\n\n")
        f.write(f"#define N_FEATURES {len(coef)}\n\n")
        f.write(f"static const float RIDGE_INTERCEPT = {intercept}f;\n\n")
        f.write("static const float RIDGE_COEF[] = {\n")
        for i, v in enumerate(coef):
            f.write(f"{v}f")
            if i != len(coef) - 1:
                f.write(", ")
        f.write("\n};\n\n")
        f.write("""static inline int ridge_predict(const float* x) {
    float score = RIDGE_INTERCEPT;
    for (int i = 0; i < N_FEATURES; i++)
        score += x[i] * RIDGE_COEF[i];
    return score > 0.0f ? 1 : 0;
}

#endif /* RIDGE_WEIGHTS_H */
""")

    log.info(f"✔ Ridge header: {path}  ({len(coef)} features)")


# ─────────────────────────────────────────────
def export_calibration_dataset(dataset_path, n_tp, out_dir, n_per_class=200):
    data  = np.load(dataset_path)
    X     = data["X"].astype(np.float32)
    y     = data["y"]

    rng  = np.random.default_rng(42)
    idx0 = rng.choice(np.where(y == 0)[0], min(n_per_class, (y==0).sum()), replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], min(n_per_class, (y==1).sum()), replace=False)
    idx  = np.concatenate([idx0, idx1])
    rng.shuffle(idx)

    X_cal = X[idx]                                    # (N, 1000)
    X_4d  = X_cal[:, np.newaxis, np.newaxis, :]       # (N, 1, 1, 1000) ← FIX ST Edge AI

    path = os.path.join(out_dir, "calibration_msrcfe.npz")
    np.savez(path, input=X_4d)

    log.info(f"✔ Calibration dataset: {path}")
    log.info(f"  Shape: {X_4d.shape}  (label=0: {len(idx0)}, label=1: {len(idx1)})")


# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model",   help="msrcfe_bundle.pkl")
    parser.add_argument("dataset", help="arc_dataset_train.npz")
    parser.add_argument("--out",   default="export_msrcfe")
    parser.add_argument("--n-cal", type=int, default=200,
                        help="Campioni per classe nel dataset di calibrazione")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── LOAD ──
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    log.info(f"Model type: {type(model)}")
    if isinstance(model, dict):
        log.info(f"Model keys: {list(model.keys())}")

    data = np.load(args.dataset)
    X    = data["X"]
    n_tp = X.shape[-1]
    log.info(f"Dataset: {X.shape}  →  n_tp={n_tp}")

    # ── LOAD MSRCFE + RIDGE ──
    msrcfe = MSRCFEFeatureExtractor()
    msrcfe.load_state_dict(model["feature_extractor_state_dict"])
    clf = model["ridge"]

    # ── EXPORT ──
    log.info("\n── Export MSRCFE ONNX ──")
    n_features = export_msrcfe_onnx(msrcfe, n_tp, args.out)

    log.info("\n── Export Ridge Header ──")
    export_ridge_header(clf, args.out)

    log.info("\n── Calibration Dataset ──")
    export_calibration_dataset(args.dataset, n_tp, args.out, args.n_cal)

    # ── SANITY CHECK: n_features del Ridge deve matchare MSRCFE ──
    ridge_n = clf.coef_.shape[-1]
    if n_features != ridge_n:
        log.error(f"✘ MISMATCH: MSRCFE produce {n_features} features, Ridge si aspetta {ridge_n}!")
    else:
        log.info(f"\n✔ Feature match: MSRCFE={n_features} == Ridge={ridge_n}")

    log.info("\n✔ EXPORT COMPLETATO")
    log.info(f"  Output: {args.out}/")
    log.info(f"    msrcfe.onnx")
    log.info(f"    ridge_weights.h")
    log.info(f"    calibration_msrcfe.npz")


if __name__ == "__main__":
    main()