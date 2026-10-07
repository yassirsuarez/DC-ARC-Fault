#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisi_test_set_v1.py - Tabelle 3.1 e 3.2, cross-fitting e bootstrap per esperimento
=====================================================================================

Riproduce, SENZA riaddestrare, le analisi della sezione 3.6:

  1. Tabella 3.1  : Accuratezza, DR, FPR, FN, FP, ROC-AUC dei 4 modelli
                    (InceptionTime e MS-RCFE+Ridge, FP32 e INT8) a soglia fissata a priori
                    (0.5 sulla probabilita' di arco; 0 sullo score del Ridge).
                    Aggiunge concordanza FP32/INT8 e concentrazione degli errori per esperimento.
  2. Tabella 3.2  : sensibilita' di InceptionTime alla soglia (default 0.50, 0.25, 0.10).
  3. Cross-fitting: 30 suddivisioni del test set in due meta' disgiunte PER ESPERIMENTO;
                    soglia scelta su una meta' (max DR con FPR <= 1%), valutata sull'altra.
                    E' una stima indicativa: il test set e' riutilizzato.
  4. Bootstrap    : ricampionamento degli ESPERIMENTI (non delle finestre) con reinserimento,
                    intervalli percentile al 95% per Accuratezza, DR, FPR e per le differenze
                    appaiate tra modelli. Cattura la variabilita' di campionamento del test set,
                    NON quella di addestramento (seed, split).

Convenzioni
  - Classe 1 = arco (positivo), classe 0 = pre-arco (negativo).
  - InceptionTime: i modelli ONNX restituiscono logit; p(arco) = softmax(logit)[1].
    p > 0.5  <=>  logit1 - logit0 > 0  (a parita' di punteggio vince la classe 0, come argmax).
  - MS-RCFE: score = features @ ridge_coef + ridge_intercept (Ridge ricostruito da .npy);
    classe 1 se score > 0 (come RidgeClassifier.predict).
  - Dev. std.: ddof=1 (deviazione standard campionaria tra gli split).

Uso (dalla radice del repository o con --repo):
    python analisi_test_set_v1.py --repo /percorso/DC-ARC-Fault --out risultati_analisi
    python analisi_test_set_v1.py --use-cache      # riusa gli score gia' calcolati
    python analisi_test_set_v1.py --n 600          # prova rapida su un sottoinsieme

Dipendenze: numpy, onnxruntime, scikit-learn (tqdm facoltativo).
Nessuna dipendenza da torch.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v1"

# --------------------------------------------------------------------------------------
# Percorsi relativi alla radice del repository
# --------------------------------------------------------------------------------------
TEST_NPZ = "dataset/dataset_new/arc_dataset_test.npz"
TEST_META = "dataset/dataset_new/arc_dataset_meta_test.csv"
MODELLI = {
    "inc_fp32": "script/training/inception_time/results/inceptiontime.onnx",
    "inc_int8": "script/modelli_quantizzati/Inception/"
                "inceptiontime_PerChannel_quant_calibration_data_npz_1.onnx",
    "ms_fp32": "script/training/msrcfe/export_msrcfe/msrcfe.onnx",
    "ms_int8": "script/modelli_quantizzati/msrcfe/"
               "msrcfe_PerChannel_quant_calibration_msrcfe_npz_1.onnx",
}
RIDGE_COEF = "script/training/msrcfe/results/ridge_coef.npy"
RIDGE_INTERCEPT = "script/training/msrcfe/results/ridge_intercept.npy"

ETICHETTE = {
    "inc_fp32": "InceptionTime (FP32)",
    "inc_int8": "InceptionTime (INT8)",
    "ms_fp32": "MS-RCFE + Ridge (FP32)",
    "ms_int8": "MS-RCFE + Ridge (INT8)",
}
ORDINE = ["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"]

# Valori di riferimento dichiarati nel riepilogo di revisione (usati SOLO come controllo
# di regressione quando si analizza il test set completo; non entrano nei calcoli).
RIF_FN_FP = {"inc_fp32": (13, 16), "inc_int8": (70, 4), "ms_fp32": (23, 3), "ms_int8": (38, 3)}
RIF_AUC = {"inc_fp32": 99.98, "inc_int8": 99.97, "ms_fp32": 99.98, "ms_int8": 99.98}
RIF_CONC = {"inc": 98.98, "ms": 99.79}
RIF_ERR_ESP = {"inc_fp32": (29, 12), "inc_int8": (74, 20), "ms_fp32": (26, 9), "ms_int8": (41, 16)}
RIF_T32_ERRORI = {("fp32", 0.10): 71, ("int8", 0.10): 45, ("fp32", 0.50): 29, ("int8", 0.50): 74}
RIF_CROSSFIT = {
    "fp32": dict(thr_med=0.41, thr_rng=(0.19, 0.82), dr=(99.75, 0.20), fpr=(0.92, 0.57),
                 dr0=(99.71, 0.15), fpr0=(0.63, 0.23)),
    "int8": dict(thr_med=0.03, thr_rng=(0.02, 0.05), dr=(99.69, 0.23), fpr=(0.72, 0.47),
                 dr0=(98.39, 0.60), fpr0=(0.13, 0.15)),
}

LOG = []


def P(testo=""):
    print(testo, flush=True)
    LOG.append(str(testo))


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()[:12]


def progresso(it, totale, descr):
    try:
        from tqdm import tqdm
        return tqdm(it, total=totale, desc=descr)
    except ImportError:
        return it


def trova_repo(arg_repo):
    candidati = []
    if arg_repo:
        candidati.append(Path(arg_repo))
    qui = Path(__file__).resolve().parent
    candidati += [qui, qui.parent, Path.cwd()]
    for c in candidati:
        if (c / TEST_NPZ).exists():
            return c
    sys.exit("Repository non trovato: usa --repo con la radice di DC-ARC-Fault "
             f"(deve contenere {TEST_NPZ}).")


# --------------------------------------------------------------------------------------
# Dati e inferenza
# --------------------------------------------------------------------------------------
def carica_test(repo, n, seed):
    d = np.load(repo / TEST_NPZ)
    X, y = d["X"], d["y"].astype(int)
    with open(repo / TEST_META, newline="", encoding="utf-8") as f:
        righe = list(csv.DictReader(f))
    if len(righe) != len(y):
        sys.exit(f"Metadati ({len(righe)}) e test set ({len(y)}) non allineati.")
    if not np.array_equal(np.array([int(r["label"]) for r in righe]), y):
        sys.exit("Le etichette dei metadati non coincidono con quelle del test set.")
    chiavi = np.array([r["exp_key"] for r in righe])
    idx = np.arange(len(y))
    if n is not None and n < len(y):
        idx = np.sort(np.random.default_rng(seed).choice(len(y), n, replace=False))
    return X[idx], y[idx], chiavi[idx], idx


def apri_sessione(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def score_inception(sess, X, batch=64):
    """Differenza dei logit (classe 1 - classe 0), float64."""
    nome = sess.get_inputs()[0].name
    out = np.empty(len(X), dtype=np.float64)
    for i in progresso(range(0, len(X), batch), (len(X) + batch - 1) // batch, "Inception"):
        xb = X[i:i + batch].astype(np.float32).reshape(-1, 1, X.shape[1])
        lg = sess.run(None, {nome: xb})[0].astype(np.float64)
        out[i:i + batch] = lg[:, 1] - lg[:, 0]
    return out


def score_msrcfe(sess, X, coef, intercetta):
    """Score del Ridge sulle 576 feature (estrattore ONNX a batch statico 1)."""
    nome = sess.get_inputs()[0].name
    out = np.empty(len(X), dtype=np.float64)
    for i in progresso(range(len(X)), len(X), "MS-RCFE"):
        f = sess.run(None, {nome: X[i].astype(np.float32).reshape(1, 1, -1)})[0]
        out[i] = float(f.astype(np.float64).reshape(-1) @ coef + intercetta)
    return out


def calcola_score(repo, X, idx, out_dir, usa_cache):
    cache = out_dir / "score_cache.npz"
    impronte = {k: sha12(repo / p) for k, p in MODELLI.items()}
    impronte["ridge_coef"] = sha12(repo / RIDGE_COEF)
    impronte["test_npz"] = sha12(repo / TEST_NPZ)
    if usa_cache and cache.exists():
        c = np.load(cache, allow_pickle=False)
        meta = json.loads(str(c["meta"]))
        if meta["impronte"] == impronte and np.array_equal(c["idx"], idx):
            P(f"Score caricati dalla cache: {cache}")
            return {k: c[k] for k in ORDINE}, impronte
        P("Cache non coerente con modelli/dati correnti: ricalcolo.")
    coef = np.load(repo / RIDGE_COEF).astype(np.float64).reshape(-1)
    intercetta = float(np.load(repo / RIDGE_INTERCEPT).reshape(-1)[0])
    sc = {}
    for k in ORDINE:
        t0 = time.time()
        s = apri_sessione(repo / MODELLI[k])
        sc[k] = (score_inception(s, X) if k.startswith("inc")
                 else score_msrcfe(s, X, coef, intercetta))
        P(f"  {ETICHETTE[k]}: {len(X)} finestre in {time.time() - t0:.1f} s")
    np.savez(cache, idx=idx, meta=json.dumps({"impronte": impronte}), **sc)
    return sc, impronte


# --------------------------------------------------------------------------------------
# Metriche
# --------------------------------------------------------------------------------------
def confusione(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


def metriche(y, pred):
    tp, fn, tn, fp = confusione(y, pred)
    return dict(tp=tp, fn=fn, tn=tn, fp=fp,
                acc=(tp + tn) / len(y), dr=tp / (tp + fn), fpr=fp / (tn + fp))


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p / (1.0 - p))


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------------------
# 1. Tabella 3.1
# --------------------------------------------------------------------------------------
def tabella_3_1(y, chiavi, sc, completo, out_dir):
    from sklearn.metrics import roc_auc_score
    pred = {k: (sc[k] > 0).astype(int) for k in ORDINE}
    righe, esp_errori = [], {}
    P("\n" + "=" * 100)
    P(f"TABELLA 3.1 - {len(y)} finestre ({int((y == 1).sum())} arco, {int((y == 0).sum())} pre-arco), "
      f"{len(set(chiavi))} esperimenti; soglia 0.5 (Ridge: 0)")
    P("=" * 100)
    P(f"{'Modello':<26}{'Acc.':>9}{'DR':>9}{'FPR':>9}{'FN':>6}{'FP':>6}{'AUC':>9}")
    for k in ORDINE:
        m = metriche(y, pred[k])
        auc = roc_auc_score(y, sc[k])
        righe.append([ETICHETTE[k], f"{100*m['acc']:.2f}", f"{100*m['dr']:.2f}",
                      f"{100*m['fpr']:.2f}", m["fn"], m["fp"], f"{100*auc:.2f}"])
        P(f"{ETICHETTE[k]:<26}{pct(m['acc']):>9}{pct(m['dr']):>9}{pct(m['fpr']):>9}"
          f"{m['fn']:>6}{m['fp']:>6}{pct(auc):>9}")
        errore = pred[k] != y
        esp_errori[k] = {e: int(errore[chiavi == e].sum()) for e in sorted(set(chiavi))}
    scrivi_csv(out_dir / "tabella_3_1.csv",
               ["Modello", "Accuratezza_%", "DR_%", "FPR_%", "FN", "FP", "ROC_AUC_%"], righe)

    P("\nConcordanza FP32/INT8 (stessa classe predetta) e direzione dei cambiamenti")
    for fam, a, b in (("inc", "inc_fp32", "inc_int8"), ("ms", "ms_fp32", "ms_int8")):
        conc = (pred[a] == pred[b]).mean()
        ok_a = pred[a] == y
        ok_b = pred[b] == y
        peggiora = int((ok_a & ~ok_b).sum())
        migliora = int((~ok_a & ok_b).sum())
        P(f"  {fam.upper():<4} concordanza {pct(conc)}  finestre cambiate {int((pred[a] != pred[b]).sum())}"
          f"  (FP32 giusta -> INT8 errata: {peggiora}; FP32 errata -> INT8 giusta: {migliora})")

    P("\nConcentrazione degli errori per esperimento (soglia 0.5)")
    P(f"  {'Modello':<26}{'Errori':>8}{'Esp. con errori':>17}{'Nei 3 peggiori':>16}{'Nel peggiore':>14}")
    righe_c = []
    for k in ORDINE:
        v = sorted(esp_errori[k].values(), reverse=True)
        tot, n_esp = sum(v), sum(1 for x in v if x > 0)
        P(f"  {ETICHETTE[k]:<26}{tot:>8}{n_esp:>17}{sum(v[:3]):>16}{v[0]:>14}")
        righe_c.append([ETICHETTE[k], tot, n_esp, sum(v[:3]), v[0]])
    scrivi_csv(out_dir / "concentrazione_errori.csv",
               ["Modello", "Errori", "Esperimenti_con_errori", "Errori_nei_3_peggiori",
                "Errori_nel_peggiore"], righe_c)
    scrivi_csv(out_dir / "errori_per_esperimento.csv", ["exp_key"] + ORDINE,
               [[e] + [esp_errori[k][e] for k in ORDINE] for e in sorted(set(chiavi))])

    if completo:
        P("\nControllo di regressione rispetto al riepilogo di revisione (test set completo)")
        for k in ORDINE:
            m = metriche(y, pred[k])
            auc = round(100 * roc_auc_score(y, sc[k]), 2)
            tot_err = m["fn"] + m["fp"]
            n_esp = sum(1 for x in esp_errori[k].values() if x > 0)
            ok = ((m["fn"], m["fp"]) == RIF_FN_FP[k] and abs(auc - RIF_AUC[k]) < 0.006
                  and (tot_err, n_esp) == RIF_ERR_ESP[k])
            P(f"  {ETICHETTE[k]:<26}{'COINCIDE' if ok else 'DIFFERISCE'}  "
              f"(FN/FP {m['fn']}/{m['fp']} vs {RIF_FN_FP[k][0]}/{RIF_FN_FP[k][1]}; "
              f"AUC {auc:.2f} vs {RIF_AUC[k]:.2f}; errori/esp. {tot_err}/{n_esp} vs "
              f"{RIF_ERR_ESP[k][0]}/{RIF_ERR_ESP[k][1]})")
        for fam, a, b in (("inc", "inc_fp32", "inc_int8"), ("ms", "ms_fp32", "ms_int8")):
            c = round(100 * (pred[a] == pred[b]).mean(), 2)
            P(f"  Concordanza {fam.upper():<4}{'COINCIDE' if abs(c - RIF_CONC[fam]) < 0.006 else 'DIFFERISCE'}"
              f"  ({c:.2f} vs {RIF_CONC[fam]:.2f})")
    return pred


# --------------------------------------------------------------------------------------
# 2. Tabella 3.2
# --------------------------------------------------------------------------------------
def tabella_3_2(y, sc, soglie, completo, out_dir):
    P("\n" + "=" * 100)
    P("TABELLA 3.2 - Sensibilita' di InceptionTime alla soglia (probabilita' di arco)")
    P("=" * 100)
    P(f"{'Versione':<10}{'Soglia':>8}{'DR':>9}{'FPR':>9}{'FN':>6}{'FP':>6}{'Errori':>8}")
    righe = []
    for ver, k in (("FP32", "inc_fp32"), ("INT8", "inc_int8")):
        for t in soglie:
            m = metriche(y, (sc[k] > logit(t)).astype(int))
            err = m["fn"] + m["fp"]
            righe.append([ver, f"{t:.2f}", f"{100*m['dr']:.2f}", f"{100*m['fpr']:.2f}",
                          m["fn"], m["fp"], err])
            extra = ""
            if completo and (ver.lower(), round(t, 2)) in RIF_T32_ERRORI:
                att = RIF_T32_ERRORI[(ver.lower(), round(t, 2))]
                extra = f"   [riferimento errori {att}: {'COINCIDE' if att == err else 'DIFFERISCE'}]"
            P(f"{ver:<10}{t:>8.2f}{pct(m['dr']):>9}{pct(m['fpr']):>9}{m['fn']:>6}{m['fp']:>6}{err:>8}{extra}")
    scrivi_csv(out_dir / "tabella_3_2.csv",
               ["Versione", "Soglia", "DR_%", "FPR_%", "FN", "FP", "Errori"], righe)


# --------------------------------------------------------------------------------------
# 3. Cross-fitting per esperimento
# --------------------------------------------------------------------------------------
def crossfitting(y, chiavi, d, n_split, seed, fpr_max, tie, passo=0.01):
    """Per ogni split: soglia (griglia 0.01..0.99) scelta sulla meta' A, valutata sulla meta' B."""
    rng = np.random.default_rng(seed)
    esp = np.unique(chiavi)
    cod = np.searchsorted(esp, chiavi)
    n_esp = len(esp)
    griglia = np.round(np.arange(passo, 1.0 - passo / 2, passo), 4)
    lt = logit(griglia)
    pos, neg = y == 1, y == 0
    out = []
    for s in range(n_split):
        perm = rng.permutation(n_esp)
        in_a = np.zeros(n_esp, bool)
        in_a[perm[: n_esp // 2]] = True
        a, b = in_a[cod], ~in_a[cod]
        dr_a = (d[a & pos][:, None] > lt).mean(0)
        fpr_a = (d[a & neg][:, None] > lt).mean(0)
        ammessi = np.where(fpr_a <= fpr_max)[0]
        if len(ammessi) == 0:                      # nessuna soglia rispetta il vincolo
            j, fallback = int(np.argmin(fpr_a)), 1
        else:
            migliori = ammessi[dr_a[ammessi] >= dr_a[ammessi].max() - 1e-12]
            j, fallback = int(migliori[-1] if tie == "highest" else migliori[0]), 0
        dr_b = float((d[b & pos] > lt[j]).mean())
        fpr_b = float((d[b & neg] > lt[j]).mean())
        out.append(dict(split=s, soglia=float(griglia[j]), dr=dr_b, fpr=fpr_b,
                        dr0=float((d[b & pos] > 0).mean()), fpr0=float((d[b & neg] > 0).mean()),
                        n_esp_a=int(in_a.sum()), fallback=fallback))
    return out


def riassunto_crossfit(nome, r, completo):
    thr = np.array([x["soglia"] for x in r])
    f = lambda k: np.array([x[k] for x in r]) * 100
    P(f"\n  {nome}: soglia mediana {np.median(thr):.2f} (min-max {thr.min():.2f}-{thr.max():.2f}; "
      f"p5-p95 {np.percentile(thr, 5):.2f}-{np.percentile(thr, 95):.2f}); "
      f"split senza soglia ammissibile: {sum(x['fallback'] for x in r)}")
    P(f"    soglia ricalibrata: DR {f('dr').mean():.2f} +/- {f('dr').std(ddof=1):.2f}   "
      f"FPR {f('fpr').mean():.2f} +/- {f('fpr').std(ddof=1):.2f}")
    P(f"    soglia 0.5        : DR {f('dr0').mean():.2f} +/- {f('dr0').std(ddof=1):.2f}   "
      f"FPR {f('fpr0').mean():.2f} +/- {f('fpr0').std(ddof=1):.2f}")
    if completo:
        ref = RIF_CROSSFIT[nome.lower()]
        P(f"    [riferimento riepilogo] soglia {ref['thr_med']:.2f} ({ref['thr_rng'][0]:.2f}-{ref['thr_rng'][1]:.2f}); "
          f"DR {ref['dr'][0]:.2f}+/-{ref['dr'][1]:.2f}, FPR {ref['fpr'][0]:.2f}+/-{ref['fpr'][1]:.2f}; "
          f"a 0.5: DR {ref['dr0'][0]:.2f}+/-{ref['dr0'][1]:.2f}, FPR {ref['fpr0'][0]:.2f}+/-{ref['fpr0'][1]:.2f}")
    return [nome, f"{np.median(thr):.2f}", f"{thr.min():.2f}", f"{thr.max():.2f}",
            f"{f('dr').mean():.2f}", f"{f('dr').std(ddof=1):.2f}",
            f"{f('fpr').mean():.2f}", f"{f('fpr').std(ddof=1):.2f}",
            f"{f('dr0').mean():.2f}", f"{f('dr0').std(ddof=1):.2f}",
            f"{f('fpr0').mean():.2f}", f"{f('fpr0').std(ddof=1):.2f}"]


# --------------------------------------------------------------------------------------
# 4. Bootstrap per esperimento
# --------------------------------------------------------------------------------------
def bootstrap_esperimenti(y, chiavi, pred, n_boot, seed, out_dir):
    esp = np.unique(chiavi)
    cod = np.searchsorted(esp, chiavi)
    n_esp = len(esp)
    cont = {}
    for k in ORDINE:
        c = np.zeros((n_esp, 4))
        for j, mask in enumerate(((pred[k] == 1) & (y == 1), (pred[k] == 0) & (y == 1),
                                  (pred[k] == 0) & (y == 0), (pred[k] == 1) & (y == 0))):
            c[:, j] = np.bincount(cod[mask], minlength=n_esp)
        cont[k] = c
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n_esp, np.full(n_esp, 1.0 / n_esp), size=n_boot).astype(float)

    def tre(k):
        t = W @ cont[k]
        tp, fn, tn, fp = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            return dict(acc=(tp + tn) / t.sum(1), dr=tp / (tp + fn), fpr=fp / (tn + fp))

    B = {k: tre(k) for k in ORDINE}
    ci = lambda v: np.nanpercentile(v, [2.5, 97.5]) * 100
    P("\n" + "=" * 100)
    P(f"BOOTSTRAP PER ESPERIMENTO - {n_boot} repliche, {n_esp} esperimenti, intervallo percentile 95%")
    P("=" * 100)
    P(f"{'Modello':<26}{'Accuratezza':>22}{'DR':>22}{'FPR':>22}")
    righe = []
    for k in ORDINE:
        m = metriche(y, pred[k])
        celle = []
        for nome in ("acc", "dr", "fpr"):
            lo, hi = ci(B[k][nome])
            celle.append(f"{100*m[nome]:.2f} [{lo:.2f}, {hi:.2f}]")
            righe.append([ETICHETTE[k], nome, f"{100*m[nome]:.2f}", f"{lo:.2f}", f"{hi:.2f}"])
        P(f"{ETICHETTE[k]:<26}{celle[0]:>22}{celle[1]:>22}{celle[2]:>22}")
    scrivi_csv(out_dir / "bootstrap_intervalli.csv",
               ["Modello", "Metrica", "Stima_%", "IC95_inf_%", "IC95_sup_%"], righe)

    P("\nDifferenze appaiate (stesse repliche), punti percentuali; IC95 e quota di repliche con diff. > 0")
    coppie = [("InceptionTime INT8 - FP32", "inc_int8", "inc_fp32"),
              ("MS-RCFE INT8 - FP32", "ms_int8", "ms_fp32"),
              ("FP32: MS-RCFE - InceptionTime", "ms_fp32", "inc_fp32"),
              ("INT8: MS-RCFE - InceptionTime", "ms_int8", "inc_int8")]
    P(f"{'Confronto':<32}{'Metrica':>8}{'Diff.':>9}{'IC95':>20}{'P(diff>0)':>11}")
    righe_d = []
    for etic, a, b in coppie:
        ma, mb = metriche(y, pred[a]), metriche(y, pred[b])
        for nome in ("acc", "dr", "fpr"):
            diff = 100 * (ma[nome] - mb[nome])
            v = (B[a][nome] - B[b][nome]) * 100
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            q = float(np.nanmean(v > 0))
            righe_d.append([etic, nome, f"{diff:.2f}", f"{lo:.2f}", f"{hi:.2f}", f"{q:.3f}"])
            P(f"{etic:<32}{nome:>8}{diff:>+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}{q:>11.3f}")
    scrivi_csv(out_dir / "bootstrap_differenze.csv",
               ["Confronto", "Metrica", "Diff_pp", "IC95_inf_pp", "IC95_sup_pp", "P_diff_maggiore_0"],
               righe_d)


# --------------------------------------------------------------------------------------
def scrivi_csv(path, intestazione, righe):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(intestazione)
        w.writerows(righe)


def main():
    ap = argparse.ArgumentParser(description="Tabelle 3.1/3.2, cross-fitting e bootstrap (senza riaddestrare).")
    ap.add_argument("--repo", help="radice del repository DC-ARC-Fault")
    ap.add_argument("--out", default="risultati_analisi", help="cartella di output")
    ap.add_argument("--n", type=int, default=None, help="usa solo n finestre casuali (prova rapida)")
    ap.add_argument("--seed", type=int, default=42, help="seed per sottocampionamento e bootstrap")
    ap.add_argument("--use-cache", action="store_true", help="riusa gli score salvati")
    ap.add_argument("--soglie", type=float, nargs="+", default=[0.50, 0.25, 0.10])
    ap.add_argument("--n-split", type=int, default=30, help="split del cross-fitting")
    ap.add_argument("--crossfit-seed", type=int, default=0)
    ap.add_argument("--fpr-max", type=float, default=0.01, help="vincolo FPR del cross-fitting")
    ap.add_argument("--tie", choices=["highest", "lowest"], default="highest",
                    help="a parita' di DR sceglie la soglia piu' alta (default) o piu' bassa")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    repo = trova_repo(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import onnxruntime, sklearn
    P(f"analisi_test_set {VERSIONE} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    P(f"numpy {np.__version__}, onnxruntime {onnxruntime.__version__}, scikit-learn {sklearn.__version__}, "
      f"Python {platform.python_version()}")

    X, y, chiavi, idx = carica_test(repo, args.n, args.seed)
    completo = args.n is None or args.n >= 7161
    P(f"Test set: {len(y)} finestre, {len(set(chiavi))} esperimenti"
      f"{'' if completo else '  [SOTTOINSIEME: i controlli di regressione sono disattivati]'}")
    sc, impronte = calcola_score(repo, X, idx, out_dir, args.use_cache)
    P("Impronte (sha256, 12 caratteri): " + ", ".join(f"{k}={v}" for k, v in impronte.items()))

    pred = tabella_3_1(y, chiavi, sc, completo, out_dir)
    tabella_3_2(y, sc, args.soglie, completo, out_dir)

    P("\n" + "=" * 100)
    P(f"CROSS-FITTING PER ESPERIMENTO - {args.n_split} split 50/50, criterio: max DR con FPR <= "
      f"{100*args.fpr_max:.0f}% (griglia 0.01-0.99, parita' -> soglia {args.tie}); stima indicativa")
    P("=" * 100)
    righe_cf, righe_split = [], []
    for nome, k in (("FP32", "inc_fp32"), ("INT8", "inc_int8")):
        r = crossfitting(y, chiavi, sc[k], args.n_split, args.crossfit_seed, args.fpr_max, args.tie)
        righe_cf.append(riassunto_crossfit(nome, r, completo))
        righe_split += [[nome, x["split"], f"{x['soglia']:.2f}", f"{100*x['dr']:.4f}",
                         f"{100*x['fpr']:.4f}", f"{100*x['dr0']:.4f}", f"{100*x['fpr0']:.4f}",
                         x["fallback"]] for x in r]
    scrivi_csv(out_dir / "crossfitting_riassunto.csv",
               ["Versione", "Soglia_mediana", "Soglia_min", "Soglia_max", "DR_medio_%", "DR_std",
                "FPR_medio_%", "FPR_std", "DR_a_0.5_%", "DR_0.5_std", "FPR_a_0.5_%", "FPR_0.5_std"],
               righe_cf)
    scrivi_csv(out_dir / "crossfitting_split.csv",
               ["Versione", "Split", "Soglia", "DR_%", "FPR_%", "DR_a_0.5_%", "FPR_a_0.5_%", "Fallback"],
               righe_split)

    if args.n_boot > 0:
        bootstrap_esperimenti(y, chiavi, pred, args.n_boot, args.seed, out_dir)

    P("\nLimiti: valori ottenuti con onnxruntime su CPU (non sul target); singolo split di training e "
      "singolo run per modello; cross-fitting e bootstrap riutilizzano il test set e non stimano la "
      "variabilita' di addestramento.")
    (out_dir / "report_analisi.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    P(f"\nFile scritti in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
