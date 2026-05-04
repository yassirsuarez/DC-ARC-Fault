#!/usr/bin/env python3
"""
export_model.py
===============
Esporta in ONNX e header C i tre modelli addestrati per deployment
su STM32H7 via X-CUBE-AI.

Modelli supportati
------------------
1. MultiRocketHydraClassifier  (model_multirockethydra.pkl)
   Pipeline: MultiRocket → StandardScaler
             HydraTransformer → _SparseScaler
             RidgeClassifierCV
   Feature totali Ridge: 56.896  (Hydra 7.168 + MultiRocket 49.728)

2. HydraClassifier  (model_hydra.pkl)
   Pipeline sklearn: HydraTransformer → _SparseScaler → RidgeClassifierCV
   Feature totali Ridge: 12.288

3. InceptionTimeClassifier  (model_inceptiontime.pkl)
   Ensemble di 5 IndividualInceptionClassifier (Keras/TF)
   Export: ogni rete Keras → ONNX via tf2onnx, poi media dei logit

Pipeline inferenza STM32H7
--------------------------
MultiRocketHydra:
  signal → [MultiRocket C] → [StandardScaler] → rocket_features
         → [Hydra ONNX]    → [SparseScaler incluso nel wrapper] → hydra_features
         → concat → [Ridge header C]  → 0/1

Hydra:
  signal → [Hydra ONNX] → [SparseScaler incluso nel wrapper] → features
         → [Ridge header C] → 0/1

InceptionTime:
  signal → [inception_i.onnx  ×5] → media logit → argmax → 0/1

OPSET aggiornati per ST Edge AI:
  - Hydra ONNX:       opset 11 → 13 → 17 (prova in ordine crescente)
  - Ridge ONNX:       opset 17 fisso (skl2onnx)
  - InceptionTime:    opset 13 (tf2onnx)
  - Ridge NON va quantizzato con ST Edge AI — usare ridge_weights.h in C

Uso:
  python export_model.py multirockethydra  model_multirockethydra.pkl  dataset.npz  --out ./export_mrh
  python export_model.py hydra             model_hydra.pkl             dataset.npz  --out ./export_hydra
  python export_model.py inceptiontime     model_inceptiontime.pkl     dataset.npz  --out ./export_it

Requisiti comuni:
  pip install aeon torch skl2onnx onnx onnxruntime

Requisiti aggiuntivi per InceptionTime:
  pip install tensorflow tf2onnx
"""

import os
import sys
import argparse
import logging
import pickle
import warnings
import io
warnings.filterwarnings("ignore")

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── opset compatibili con ST Edge AI ─────────────────────────────────────────
# ST Edge AI supporta ufficialmente fino a opset 21.
# Hydra: prova in ordine crescente fino a trovare quello che funziona.
# Ridge (skl2onnx): usa opset 17 fisso.
# InceptionTime (tf2onnx): usa opset 13 fisso.
# NON usare opset 22+ — onnxruntime non lo supporta ancora.
HYDRA_OPSETS   = (11, 13, 17)   # MODIFICATO: era (13, 16) — aggiunto 17
RIDGE_OPSET    = 17             # MODIFICATO: era implicito in skl2onnx default
INCEPTION_OPSET = 13            # invariato — tf2onnx con opset 13


# ══════════════════════════════════════════════════════════════════════════════
# Utilità condivise
# ══════════════════════════════════════════════════════════════════════════════

def load_sample(dataset_path: str, n: int = 4,
                n_timepoints: int = None) -> tuple:
    """
    Carica n campioni dal dataset .npz.
    Se n_timepoints è specificato tronca le serie alla lunghezza
    vista durante il training (ricavata da model.metadata_).
    """
    data = np.load(dataset_path)
    X = data["X"][:n].astype(np.float32)
    y = data["y"][:n]
    if n_timepoints is not None and X.shape[-1] != n_timepoints:
        log.info("  Dataset lunghezza=%d, training su %d → troncamento",
                 X.shape[-1], n_timepoints)
        X = X[..., :n_timepoints]
    log.info("Campioni test: %d  shape: %s", n, X.shape)
    return X, y


def get_train_n_timepoints(model) -> int:
    """Ricava la lunghezza serie di training da model.metadata_."""
    meta = getattr(model, "metadata_", {})
    n_tp = meta.get("n_timepoints", None)
    if n_tp is not None:
        log.info("  Lunghezza serie training (metadata_): %d", n_tp)
    return n_tp


def _write_array_c(f, name: str, arr: np.ndarray):
    """Scrive un array 1-D o 2-D come costante C in un header già aperto."""
    arr = arr.astype(np.float32)
    dtype = "float"
    if arr.ndim == 1:
        f.write(f"\nstatic const {dtype} {name}[{len(arr)}] = {{\n  ")
        vals  = [f"{v:.8f}f" for v in arr]
        lines = [", ".join(vals[i:i+8]) for i in range(0, len(vals), 8)]
        f.write(",\n  ".join(lines))
        f.write("\n};\n")
    elif arr.ndim == 2:
        f.write(f"\nstatic const {dtype} {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n")
        for row in arr:
            vals = [f"{v:.8f}f" for v in row]
            f.write("  {" + ", ".join(vals) + "},\n")
        f.write("};\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MultiRocket (pesi + scaler → header C)
# ══════════════════════════════════════════════════════════════════════════════

def export_multirocket(model, out_dir: str) -> bool:
    log.info("")
    log.info("=== EXPORT MULTIROCKET ===")
    mr = getattr(model, "_transform_multirocket", None)
    if mr is None:
        log.error("  _transform_multirocket non trovato")
        return False

    attrs = {}
    for attr in sorted(dir(mr)):
        if attr.startswith("__"):
            continue
        val = getattr(mr, attr, None)
        if val is None or callable(val):
            continue
        if hasattr(val, "shape"):
            log.info("    %s: shape=%s dtype=%s", attr, val.shape, val.dtype)
            attrs[attr] = val
        elif isinstance(val, (int, float, bool)):
            log.info("    %s: %s", attr, val)
            attrs[attr] = val

    if not attrs:
        log.warning("  Nessun attributo estratto da MultiRocket")
        return False

    npz_path = os.path.join(out_dir, "multirocket_weights.npz")
    np.savez(npz_path, **{k: np.array(v) for k, v in attrs.items()
                          if hasattr(v, "__len__") or isinstance(v, (int, float))})
    log.info("  Pesi .npz: %s", npz_path)

    # Header C rocket
    h_path = os.path.join(out_dir, "rocket_weights.h")
    with open(h_path, "w") as f:
        f.write("/* rocket_weights.h — pesi MultiRocket per STM32\n")
        f.write(" * Generato automaticamente da export_model.py\n */\n\n")
        f.write("#ifndef ROCKET_WEIGHTS_H\n#define ROCKET_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        for name, val in attrs.items():
            arr = np.array(val)
            if arr.ndim == 0:
                f.write(f"#define ROCKET_{name.upper()} {int(arr)}\n")
            elif arr.ndim == 1:
                dtype = "float" if arr.dtype.kind == "f" else "int32_t"
                f.write(f"\n/* {name}: {arr.shape} */\n")
                f.write(f"static const {dtype} ROCKET_{name.upper()}[] = {{\n  ")
                vals  = [f"{v:.8f}f" if arr.dtype.kind == "f" else str(int(v)) for v in arr]
                lines = [", ".join(vals[i:i+8]) for i in range(0, len(vals), 8)]
                f.write(",\n  ".join(lines))
                f.write("\n};\n")
            elif arr.ndim == 2:
                dtype = "float" if arr.dtype.kind == "f" else "int32_t"
                f.write(f"\n/* {name}: {arr.shape} */\n")
                f.write(f"static const {dtype} ROCKET_{name.upper()}[{arr.shape[0]}][{arr.shape[1]}] = {{\n")
                for row in arr:
                    vals = [f"{v:.8f}f" if arr.dtype.kind == "f" else str(int(v)) for v in row]
                    f.write("  {" + ", ".join(vals) + "},\n")
                f.write("};\n")
        f.write("\n#endif /* ROCKET_WEIGHTS_H */\n")
    log.info("  Header C: %s", h_path)

    # StandardScaler
    scaler = getattr(model, "_scale_multirocket", None)
    if scaler is not None and hasattr(scaler, "mean_"):
        np.savez(os.path.join(out_dir, "scaler_multirocket.npz"),
                 mean=scaler.mean_, scale=scaler.scale_)
        h_path = os.path.join(out_dir, "scaler_multirocket.h")
        mean  = scaler.mean_.astype(np.float32)
        scale = scaler.scale_.astype(np.float32)
        n     = len(mean)
        with open(h_path, "w") as f:
            f.write("/* scaler_multirocket.h — StandardScaler\n")
            f.write(" * x_scaled[i] = (x[i] - SCALER_MEAN[i]) / SCALER_SCALE[i]\n */\n\n")
            f.write("#ifndef SCALER_MULTIROCKET_H\n#define SCALER_MULTIROCKET_H\n\n")
            f.write(f"#define SCALER_N_FEATURES {n}\n")
            _write_array_c(f, "SCALER_MEAN",  mean)
            _write_array_c(f, "SCALER_SCALE", scale)
            f.write("\n#endif /* SCALER_MULTIROCKET_H */\n")
        log.info("  Header scaler: %s", h_path)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Hydra ONNX  (usato da MultiRocketHydra e da HydraClassifier)
# ══════════════════════════════════════════════════════════════════════════════

class _HydraWrapper:
    """
    Wrapper PyTorch per HydraTransformer compatibile con export ONNX.

    Riceve X e diff_X separatamente per evitare torch.diff nel grafo.
    Il SparseScaler è fuso nel forward così da essere incluso nell'ONNX.
    """

    @staticmethod
    def build(hydra_module, sparse_scaler=None):
        import torch
        import torch.nn as nn

        epsilon = None
        mu      = None
        sigma   = None
        mask    = None

        if sparse_scaler is not None:
            epsilon = sparse_scaler.epsilon.float()
            mu      = sparse_scaler.mu.float()
            sigma   = sparse_scaler.sigma.float()
            mask_v  = sparse_scaler.mask
            mask    = bool(mask_v)

        class HydraONNXWrapper(nn.Module):
            def __init__(self, hydra, eps, mu_, sigma_, use_mask):
                super().__init__()
                self.hydra         = hydra
                self.num_dilations = hydra.num_dilations
                self.divisor       = hydra.divisor
                if eps is not None:
                    self.register_buffer("scaler_epsilon", eps)
                    self.register_buffer("scaler_mu",      mu_)
                    self.register_buffer("scaler_sigma",   sigma_)
                    self.has_scaler = True
                    self.use_mask   = use_mask
                else:
                    self.has_scaler = False

            def forward(self, X, diff_X):
                X_exp      = X.expand(-1, self.hydra.g, -1)
                diff_X_exp = diff_X.expand(-1, self.hydra.g, -1)
                results = []
                for d_idx in range(self.num_dilations):
                    for diff_idx in range(self.divisor):
                        src = X_exp if diff_idx == 0 else diff_X_exp
                        p   = self.hydra.paddings[d_idx].item()
                        d   = self.hydra.dilations[d_idx].item()
                        w   = self.hydra.W[d_idx, diff_idx]
                        out = torch.nn.functional.conv1d(
                            src, w, padding=p, dilation=d, groups=self.hydra.g
                        ).view(X.shape[0], self.hydra.h, self.hydra.k, -1)
                        results.append(out.max(dim=-1).values)
                        results.append((out > 0).float().mean(dim=-1))
                features = torch.cat(results, dim=-1).view(X.shape[0], -1)

                if self.has_scaler:
                    exponent = 0.25
                    if self.use_mask:
                        features = features ** exponent
                    features = (features - self.scaler_mu) / (
                        self.scaler_sigma + self.scaler_epsilon
                    )
                return features

        return HydraONNXWrapper(hydra_module, epsilon, mu, sigma, mask)


def export_hydra_onnx(hydra_transform, sparse_scaler, n_tp: int,
                      out_dir: str, onnx_name: str = "hydra.onnx") -> tuple:
    """
    Esporta HydraTransformer (+ SparseScaler fuso) in ONNX.

    MODIFICATO: prova opset in ordine (11, 13, 17) invece di (13, 16).
    ST Edge AI supporta fino a opset 21 — opset 17 è il più recente sicuro.
    Il Ridge NON viene esportato per ST Edge AI — usare ridge_weights.h in C.

    Ritorna (successo: bool, n_features: int).
    """
    log.info("")
    log.info("=== EXPORT HYDRA (ONNX) → %s ===", onnx_name)
    log.info("  Opset da provare: %s", HYDRA_OPSETS)

    try:
        import torch
    except ImportError:
        log.error("  torch non installato")
        return False, 0

    hydra_module = getattr(hydra_transform, "_hydra", None)
    if hydra_module is None:
        log.warning("  _hydra non trovato; attributi: %s",
                    [a for a in dir(hydra_transform) if not a.startswith("__")])
        return False, 0

    hydra_module.eval()
    dummy_X    = torch.zeros(1, 1, n_tp)
    dummy_diff = torch.zeros(1, 1, n_tp - 1)

    wrapper = _HydraWrapper.build(hydra_module, sparse_scaler)
    wrapper.eval()

    try:
        with torch.no_grad():
            out_sample = wrapper(dummy_X, dummy_diff)
        n_features = out_sample.shape[-1]
        log.info("  Forward pass OK — feature: %d", n_features)
    except Exception as e:
        log.error("  Forward pass fallito: %s", e)
        return False, 0

    onnx_path = os.path.join(out_dir, onnx_name)

    def _do_export(opset):
        torch.onnx.export(
            wrapper,
            (dummy_X, dummy_diff),
            onnx_path,
            export_params=True,
            opset_version=opset,   # MODIFICATO: variabile invece di fisso
            input_names=["input", "input_diff"],
            output_names=["hydra_features"],
            dynamic_axes={
                "input":          {0: "batch"},
                "input_diff":     {0: "batch"},
                "hydra_features": {0: "batch"},
            },
        )

    # MODIFICATO: usa HYDRA_OPSETS = (11, 13, 17) invece di (13, 16)
    for opset in HYDRA_OPSETS:
        try:
            _do_export(opset)
            size_kb = os.path.getsize(onnx_path) / 1024
            log.info("  ONNX (opset %d): %s  (%.1f KB)", opset, onnx_path, size_kb)
            log.info("  input         : (batch, 1, %d)", n_tp)
            log.info("  input_diff    : (batch, 1, %d)", n_tp - 1)
            log.info("  hydra_features: (batch, %d)", n_features)
            log.info("  Nota STM32: diff_X = X[1:] - X[:-1]")

            # Verifica con onnxruntime
            try:
                import onnxruntime as rt
                sess = rt.InferenceSession(onnx_path)
                feed = {
                    "input":      dummy_X.numpy(),
                    "input_diff": dummy_diff.numpy(),
                }
                out = sess.run(None, feed)[0]
                log.info("  Verifica onnxruntime: output shape %s  ✓", out.shape)
            except Exception as e_rt:
                log.warning("  Verifica onnxruntime: %s", e_rt)

            return True, n_features
        except Exception as e:
            log.warning("  Export opset %d fallito: %s", opset, e)

    log.error("  Export Hydra ONNX fallito su tutti gli opset %s", HYDRA_OPSETS)
    return False, 0


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Ridge (header C + ONNX)
# ══════════════════════════════════════════════════════════════════════════════

def export_ridge(clf, n_features_total: int, out_dir: str,
                 prefix: str = "") -> bool:
    """
    Esporta RidgeClassifierCV in header C e ONNX.

    IMPORTANTE: il Ridge NON va quantizzato con ST Edge AI.
    È un semplice dot product — usare direttamente ridge_weights.h in C.
    L'ONNX è generato solo per documentazione/debug.

    MODIFICATO: opset Ridge impostato a RIDGE_OPSET=17 (era default skl2onnx).
    skl2onnx con opset 22 causa l'errore 'Opset 22 is under development'.
    """
    log.info("")
    log.info("=== EXPORT RIDGE (%s) ===", prefix or "default")
    log.info("  NOTA: il Ridge NON va quantizzato in ST Edge AI — usa ridge_weights.h")

    if clf is None or not hasattr(clf, "coef_"):
        log.error("  Classificatore Ridge non trovato o non fittato")
        return False

    log.info("  coef_ shape: %s", clf.coef_.shape)
    log.info("  intercept_ : %s", clf.intercept_)
    log.info("  alpha_     : %s", getattr(clf, "alpha_", "n/a"))

    # .npy
    np.save(os.path.join(out_dir, f"{prefix}ridge_coef.npy"),      clf.coef_)
    np.save(os.path.join(out_dir, f"{prefix}ridge_intercept.npy"), clf.intercept_)

    # Header C — questo è il file da usare su STM32
    coef      = clf.coef_.flatten().astype(np.float32)
    intercept = float(clf.intercept_[0])
    n         = len(coef)
    guard     = f"{prefix.upper()}RIDGE_WEIGHTS_H"
    h_path    = os.path.join(out_dir, f"{prefix}ridge_weights.h")
    with open(h_path, "w") as f:
        f.write(f"/* {prefix}ridge_weights.h — Ridge Classifier per STM32\n")
        f.write(f" * NON quantizzare con ST Edge AI — è già C puro.\n")
        f.write(f" * score = dot(features, {prefix.upper()}RIDGE_COEF) + {prefix.upper()}RIDGE_INTERCEPT\n")
        f.write(f" * label = score > 0 ? 1 : 0\n */\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write(f"#define {prefix.upper()}RIDGE_N_FEATURES {n}\n\n")
        f.write(f"static const float {prefix.upper()}RIDGE_INTERCEPT = {intercept:.8f}f;\n")
        _write_array_c(f, f"{prefix.upper()}RIDGE_COEF", coef)
        f.write(f"\nstatic inline int {prefix}ridge_predict(const float* features) {{\n")
        f.write(f"    float score = {prefix.upper()}RIDGE_INTERCEPT;\n")
        f.write(f"    for (int i = 0; i < {prefix.upper()}RIDGE_N_FEATURES; i++)\n")
        f.write(f"        score += features[i] * {prefix.upper()}RIDGE_COEF[i];\n")
        f.write(f"    return score > 0.0f ? 1 : 0;\n}}\n\n")
        f.write(f"#endif /* {guard} */\n")
    log.info("  Header C: %s  ← usa questo su STM32", h_path)

    # ONNX via skl2onnx con opset fisso (MODIFICATO: target_opset=RIDGE_OPSET)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.linear_model import RidgeClassifier

        r = RidgeClassifier(alpha=getattr(clf, "alpha_", 1.0))
        dummy_X_fit = np.zeros((2, n_features_total), dtype=np.float32)
        dummy_y_fit = clf.classes_.astype(int)
        r.fit(dummy_X_fit, dummy_y_fit)
        r.coef_      = clf.coef_
        r.intercept_ = clf.intercept_

        # MODIFICATO: target_opset=RIDGE_OPSET=17 — evita opset 22 non supportato
        onnx_model = convert_sklearn(
            r,
            initial_types=[("float_input", FloatTensorType([None, n_features_total]))],
            target_opset=RIDGE_OPSET,
        )
        onnx_path = os.path.join(out_dir, f"{prefix}ridge.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info("  ONNX (opset %d): %s  (%.1f KB) ← solo debug, NON quantizzare",
                 RIDGE_OPSET, onnx_path, os.path.getsize(onnx_path) / 1024)
    except Exception as e:
        log.warning("  skl2onnx Ridge fallito (header C già OK): %s", e)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Verifica end-to-end (MultiRocketHydra e Hydra)
# ══════════════════════════════════════════════════════════════════════════════

def verify_pipeline(model, X_sample: np.ndarray, out_dir: str,
                    model_type: str = "multirockethydra"):
    """
    Verifica che il Ridge manuale (coef_ + intercept_) produca le stesse
    predizioni del modello originale.
    """
    log.info("")
    log.info("=== VERIFICA END-TO-END (%s) ===", model_type)

    X_3d = X_sample[:, np.newaxis, :]

    try:
        y_orig = model.predict(X_3d)
        log.info("  Predizioni originali: %s", y_orig.tolist())
    except Exception as e:
        log.error("  model.predict fallito: %s", e)
        log.warning("  Verifica saltata — controlla lunghezza serie")
        return

    try:
        if model_type == "multirockethydra":
            hydra_t  = model._transform_hydra.transform(X_3d)
            hydra_s  = model._scale_hydra.transform(hydra_t)
            rocket_t = model._transform_multirocket.transform(X_3d)
            rocket_s = model._scale_multirocket.transform(rocket_t)
            features = np.hstack([hydra_s, rocket_s]).astype(np.float32)
        else:
            ht, ss, _ = [step for _, step in model._clf.steps]
            hydra_t  = ht.transform(X_3d)
            features = ss.transform(hydra_t).astype(np.float32)

        prefix    = "mrh_" if model_type == "multirockethydra" else ""
        coef      = np.load(os.path.join(out_dir, f"{prefix}ridge_coef.npy"))
        intercept = np.load(os.path.join(out_dir, f"{prefix}ridge_intercept.npy"))
        scores    = features @ coef.T + intercept
        y_manual  = (scores.flatten() > 0).astype(int)
        match     = np.all(y_manual == y_orig)
        log.info("  Ridge manuale: %s  — predizioni: %s",
                 "✓ IDENTICO" if match else "✗ DIVERSO", y_manual.tolist())
    except Exception as e:
        log.error("  Verifica Ridge manuale fallita: %s", e)

    # Verifica Hydra ONNX con onnxruntime
    for candidate in ("hydra.onnx", "hydra_standalone.onnx"):
        onnx_path = os.path.join(out_dir, candidate)
        if not os.path.isfile(onnx_path):
            continue
        try:
            import onnxruntime as rt
            sess      = rt.InferenceSession(onnx_path)
            inp_names = [i.name for i in sess.get_inputs()]
            inp       = X_3d.astype(np.float32)
            feed      = {inp_names[0]: inp}
            if len(inp_names) > 1:
                feed[inp_names[1]] = np.diff(inp, axis=-1)
            out = sess.run(None, feed)[0]
            log.info("  %s output shape: %s  ✓", candidate, out.shape)
        except Exception as e:
            log.error("  Verifica %s fallita: %s", candidate, e)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Export InceptionTime (Keras → ONNX via tf2onnx)
# ══════════════════════════════════════════════════════════════════════════════

def export_inceptiontime(model_path: str, out_dir: str) -> bool:
    """
    Esporta ogni IndividualInceptionClassifier come ONNX separato.

    MODIFICATO: opset fisso a INCEPTION_OPSET=13 (invariato, tf2onnx).
    """
    log.info("")
    log.info("=== EXPORT INCEPTIONTIME (opset %d) ===", INCEPTION_OPSET)

    try:
        import tensorflow as tf
        import tf2onnx
        log.info("  TensorFlow %s  tf2onnx %s",
                 tf.__version__, tf2onnx.__version__)
    except ImportError as e:
        log.error("  Dipendenze mancanti: %s", e)
        log.error("  Installa: pip install tensorflow tf2onnx")
        return False

    log.info("  Caricamento modello: %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    n_classifiers = len(model.classifiers_)
    log.info("  Ensemble: %d classificatori", n_classifiers)

    exported = []
    for i, clf_i in enumerate(model.classifiers_):
        keras_model = getattr(clf_i, "model_", None)
        if keras_model is None:
            log.warning("  [%d] model_ non trovato — skip", i)
            continue

        onnx_path = os.path.join(out_dir, f"inception_{i}.onnx")
        try:
            input_shape = getattr(clf_i, "input_shape", None)
            if input_shape is None:
                try:
                    input_shape = keras_model.input_shape[1:]
                except Exception:
                    input_shape = (None, 1)

            spec = (tf.TensorSpec(
                shape=(None,) + tuple(input_shape),
                dtype=tf.float32,
                name="input"
            ),)

            # MODIFICATO: opset=INCEPTION_OPSET fisso (13)
            onnx_model_proto, _ = tf2onnx.convert.from_keras(
                keras_model,
                input_signature=spec,
                opset=INCEPTION_OPSET,
                output_path=onnx_path,
            )
            size_kb = os.path.getsize(onnx_path) / 1024
            log.info("  [%d] inception_%d.onnx  (%.1f KB)  opset=%d",
                     i, i, size_kb, INCEPTION_OPSET)
            exported.append(i)
        except Exception as e:
            log.error("  [%d] Export fallito: %s", i, e)

    if not exported:
        log.error("  Nessun modello esportato")
        return False

    # Header C ensemble
    h_path = os.path.join(out_dir, "inception_ensemble.h")
    n_exp  = len(exported)
    with open(h_path, "w") as f:
        f.write("/* inception_ensemble.h\n")
        f.write(" * Ensemble InceptionTime per STM32H7 via X-CUBE-AI\n */\n\n")
        f.write("#ifndef INCEPTION_ENSEMBLE_H\n#define INCEPTION_ENSEMBLE_H\n\n")
        f.write(f"#define INCEPTION_N_MODELS   {n_exp}\n")
        f.write(f"#define INCEPTION_N_CLASSES  {model.n_classes_}\n")
        f.write(f"#define INCEPTION_DEPTH      {model.depth}\n")
        f.write(f"#define INCEPTION_N_FILTERS  {model.n_filters}\n")
        f.write(f"#define INCEPTION_KERNEL_SIZE {model.kernel_size}\n\n")
        f.write("static const char* INCEPTION_MODEL_FILES[] = {\n")
        for i in exported:
            f.write(f'  "inception_{i}.onnx",\n')
        f.write("};\n\n")
        f.write("#endif /* INCEPTION_ENSEMBLE_H */\n")
    log.info("  Header ensemble: %s", h_path)

    return len(exported) == n_classifiers


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Funzioni di alto livello per modello
# ══════════════════════════════════════════════════════════════════════════════

def run_multirockethydra(model_path: str, dataset_path: str, out_dir: str):
    log.info("━" * 60)
    log.info("MULTI-ROCKET HYDRA CLASSIFIER")
    log.info("━" * 60)
    os.makedirs(out_dir, exist_ok=True)

    log.info("Caricamento: %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    log.info("  Tipo: %s", type(model).__name__)

    n_tp_train = get_train_n_timepoints(model)
    X_sample, _ = load_sample(dataset_path, n_timepoints=n_tp_train)
    n_tp_onnx = n_tp_train if n_tp_train is not None else X_sample.shape[-1]

    rocket_ok           = export_multirocket(model, out_dir)
    hydra_ok, n_f_hydra = export_hydra_onnx(
        model._transform_hydra,
        model._scale_hydra,
        n_tp_onnx, out_dir, "hydra.onnx"
    )

    n_feat_ridge  = model.classifier.coef_.shape[-1]
    n_feat_hydra  = model._scale_hydra.mu.shape[0]
    n_feat_rocket = model._scale_multirocket.mean_.shape[0]
    log.info("")
    log.info("  Feature Ridge totali: %d  (Hydra=%d + MultiRocket=%d)",
             n_feat_ridge, n_feat_hydra, n_feat_rocket)

    ridge_ok = export_ridge(model.classifier, n_feat_ridge, out_dir, prefix="mrh_")
    verify_pipeline(model, X_sample, out_dir, model_type="multirockethydra")

    _print_summary("MultiRocketHydra", out_dir, {
        "MultiRocket (.npz + .h + scaler)":          rocket_ok,
        "Hydra (.onnx opset≤17, SparseScaler fuso)": hydra_ok,
        "Ridge (.h C puro + .onnx debug)":            ridge_ok,
    })

    log.info("")
    log.info("  ST Edge AI: quantizza SOLO hydra.onnx con calibration_data_hydra.npz")
    log.info("  Ridge:      usa mrh_ridge_weights.h direttamente in C (NO quantizzazione)")


def run_hydra(model_path: str, dataset_path: str, out_dir: str):
    log.info("━" * 60)
    log.info("HYDRA CLASSIFIER")
    log.info("━" * 60)
    os.makedirs(out_dir, exist_ok=True)

    log.info("Caricamento: %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    log.info("  Tipo: %s", type(model).__name__)

    steps = {name: step for name, step in model._clf.steps}
    hydra_transform = steps.get("hydratransformer")
    sparse_scaler   = steps.get("_sparsescaler")
    clf             = steps.get("ridgeclassifiercv")

    if hydra_transform is None:
        log.error("  'hydratransformer' non trovato nella pipeline")
        return

    n_tp_train = get_train_n_timepoints(model)
    X_sample, _ = load_sample(dataset_path, n_timepoints=n_tp_train)
    n_tp_onnx = n_tp_train if n_tp_train is not None else X_sample.shape[-1]

    hydra_ok, _ = export_hydra_onnx(
        hydra_transform, sparse_scaler,
        n_tp_onnx, out_dir, "hydra_standalone.onnx"
    )

    n_features = clf.coef_.shape[-1]
    log.info("  Feature Ridge (da coef_): %d", n_features)

    ridge_ok = export_ridge(clf, n_features, out_dir, prefix="")
    verify_pipeline(model, X_sample, out_dir, model_type="hydra")

    _print_summary("HydraClassifier", out_dir, {
        "Hydra standalone (.onnx opset≤17, SparseScaler fuso)": hydra_ok,
        "Ridge (.h C puro + .onnx debug)":                       ridge_ok,
    })

    log.info("")
    log.info("  ST Edge AI: quantizza SOLO hydra_standalone.onnx")
    log.info("  Ridge:      usa ridge_weights.h direttamente in C (NO quantizzazione)")


def run_inceptiontime(model_path: str, dataset_path: str, out_dir: str):
    log.info("━" * 60)
    log.info("INCEPTIONTIME CLASSIFIER")
    log.info("━" * 60)
    os.makedirs(out_dir, exist_ok=True)

    ok = export_inceptiontime(model_path, out_dir)

    _print_summary("InceptionTime", out_dir, {
        "5× inception_i.onnx + inception_ensemble.h": ok,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Riepilogo
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(name: str, out_dir: str, results: dict):
    log.info("")
    log.info("=" * 60)
    log.info("RIEPILOGO  —  %s", name)
    log.info("=" * 60)
    for label, ok in results.items():
        log.info("  %-50s %s", label, "✓" if ok else "✗")
    log.info("")
    log.info("File in: %s", out_dir)
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        size  = os.path.getsize(fpath) / 1024
        log.info("  %-45s  %.1f KB", fname, size)
    log.info("")
    log.info("Prossimo passo: importare i .onnx in STM32Cube.AI (X-CUBE-AI)")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Export modelli serie temporali per STM32H7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempi d'uso:\n"
            "  python export_model.py multirockethydra model_multirockethydra.pkl arc_dataset.npz --out ./export_mrh\n"
            "  python export_model.py hydra             model_hydra.pkl            arc_dataset.npz --out ./export_hydra\n"
            "  python export_model.py inceptiontime     model_inceptiontime.pkl    arc_dataset.npz --out ./export_it\n"
            "\n"
            "Opset utilizzati:\n"
            f"  Hydra:         {HYDRA_OPSETS} (prova in ordine)\n"
            f"  Ridge:         {RIDGE_OPSET} (fisso, solo debug — NON quantizzare)\n"
            f"  InceptionTime: {INCEPTION_OPSET} (fisso)\n"
        ),
    )
    parser.add_argument(
        "model_type",
        choices=["multirockethydra", "hydra", "inceptiontime"],
        help="Tipo di modello",
    )
    parser.add_argument("model",   help="Percorso al file .pkl del modello")
    parser.add_argument("dataset", help="Percorso al file .npz del dataset")
    parser.add_argument("--out", "-o", default="./export",
                        help="Cartella di output (default: ./export)")
    args = parser.parse_args()

    for p in [args.model, args.dataset]:
        if not os.path.isfile(p):
            log.error("File non trovato: %s", p)
            sys.exit(1)

    dispatch = {
        "multirockethydra": run_multirockethydra,
        "hydra":            run_hydra,
        "inceptiontime":    run_inceptiontime,
    }
    dispatch[args.model_type](args.model, args.dataset, args.out)


if __name__ == "__main__":
    main()