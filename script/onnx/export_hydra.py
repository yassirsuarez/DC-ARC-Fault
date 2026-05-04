#!/usr/bin/env python3
"""
export_hydra_only.py
====================
Export SOLO modello Hydra per STM32H7

✔ supporta checkpoint dict:
   {
     "hydra_state_dict": ...,
     "ridge": ...
   }
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
# HYDRA ARCH (DEVE MATCHARE IL TRAINING)
# ─────────────────────────────────────────────
class HydraFeatureExtractor(nn.Module):
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
        # usato durante il training: x è (batch, T)
        x = x.unsqueeze(1)
        return self._extract(x)

    def forward_no_unsqueeze(self, x):
        # usato per ONNX export: x è già (batch, 1, T)
        return self._extract(x)

    def _extract(self, x):
        feats = []
        for conv in self.convs:
            y = torch.relu(conv(x))
            feats.append(torch.max(y, dim=-1).values)
            feats.append(torch.mean(y, dim=-1))
        return torch.cat(feats, dim=1)


# ─────────────────────────────────────────────
class HydraONNXWrapper(torch.nn.Module):
    def __init__(self, hydra):
        super().__init__()
        self.hydra = hydra

    def forward(self, x):
        return self.hydra.forward_no_unsqueeze(x)


# ─────────────────────────────────────────────
def export_hydra_onnx(hydra, n_tp, out_dir):
    import torch.onnx

    hydra.eval()

    wrapper = HydraONNXWrapper(hydra)
    wrapper.eval()

    dummy = torch.zeros(1, 1, n_tp)

    path = os.path.join(out_dir, "hydra.onnx")

    torch.onnx.export(
        wrapper,
        dummy,
        path,
        opset_version=17,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}}
    )

    log.info(f"✔ Hydra ONNX: {path}")


# ─────────────────────────────────────────────
def export_ridge_header(clf, out_dir):
    coef = clf.coef_.flatten().astype(np.float32)
    intercept = float(clf.intercept_[0])

    path = os.path.join(out_dir, "ridge_weights.h")

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

        f.write("""
static inline int ridge_predict(const float* x) {
    float score = RIDGE_INTERCEPT;
    for (int i = 0; i < N_FEATURES; i++)
        score += x[i] * RIDGE_COEF[i];
    return score > 0.0f ? 1 : 0;
}
#endif
""")

    log.info(f"✔ Ridge header: {path}")


# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("--out", default="export_hydra")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # LOAD
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print("MODEL TYPE:", type(model))
    if isinstance(model, dict):
        print("MODEL KEYS:", model.keys())

    data = np.load(args.dataset)
    X = data["X"]
    n_tp = X.shape[-1]

    # ─────────────────────────────
    # FIX CORE (IL TUO CASO)
    # ─────────────────────────────
    hydra = HydraFeatureExtractor()
    hydra.load_state_dict(model["hydra_state_dict"])

    clf = model["ridge"]

    # ── EXPORT ──
    export_hydra_onnx(hydra, n_tp, args.out)
    export_ridge_header(clf, args.out)

    log.info("\n✔ EXPORT COMPLETATO")


if __name__ == "__main__":
    main()