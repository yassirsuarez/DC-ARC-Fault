#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensibilita_etichette_v2.py - sensibilita' dei risultati ai parametri di etichettatura
=======================================================================================

L'etichetta di una finestra dipende da tre parametri fissati empiricamente in
dataset/build_dataset_new.py:

    THRESH_FRAC = 0.95   soglia = nominale x THRESH_FRAC
    CONFIRM_MS  = 300    durata minima sotto soglia per confermare t1 [ms]
    MIN_NOM_A   = 0.10   corrente nominale minima accettata [A]

Questo script ricostruisce le finestre del TEST SET dai file .mat con la stessa procedura della
pipeline (finestre di 100 ms, passo 50 ms, normalizzazione I/nominale, finestre a cavallo di t1
scartate, al massimo 10 finestre per classe e per file con seed 42) e, variando UN parametro per
volta, ricalcola le etichette e rivaluta i quattro modelli GIA' ADDESTRATI (nessun riaddestramento):
InceptionTime e MS-RCFE + Ridge, FP32 e INT8. Il caso "base" (0.95, 300 ms, 0.10 A) deve riprodurre
il test set pubblicato: lo script lo verifica confrontando, per ogni file, il numero di finestre
di ciascuna classe con arc_dataset_meta_test.csv.

Per ogni variante riporta numero di finestre, DR, FPR e accuratezza di ogni modello con intervallo
di confidenza al 95% (bootstrap per esperimento), e lo spostamento di t1 rispetto al caso base.
I modelli sono addestrati con le etichette del caso base: il risultato misura quanto le prestazioni
dipendano dalla definizione di "arco", non la robustezza di un modello riaddestrato.

Novita' della v2: nel test set alcuni nomi di file compaiono in piu' cartelle del dataset (stesso nome,
file distinti); la v1 ne leggeva solo uno per nome e il caso base risultava di 7013 finestre invece di 7161.
La v2 legge tutte le istanze (chiave "nome", "nome#2", "nome#3") e riusa gli score gia' calcolati dalla v1
(stessa cartella --out): inferenza solo sulle istanze mancanti.

Le finestre vengono valutate una sola volta (tutte le finestre dei file del test set) e gli
score sono salvati in score_finestre.npz: rilanciando con la stessa --out si salta l'inferenza.
Tempo atteso: da 10 a 40 minuti per l'inferenza (InceptionTime) al primo avvio, pochi secondi dopo.

Uso
    python sensibilita_etichette_v2.py --repo . --mat-root "<cartella dei .mat>" --out ris_sensibilita
    python sensibilita_etichette_v2.py --selftest
Richiede script/analisi_test_set_v1.py, i modelli ONNX e dataset/dataset_new/*test* nel repository.
Scrive sensibilita_etichette.csv, riepilogo_sensibilita.json (senza il percorso dei .mat) e
report.txt.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v2"
FS = 10000
WIN_N, STEP_N, NOMINAL_N = 1000, 500, 2000     # 100 ms, 50 ms, 0.2 s
MAX_WIN = 10
SEED_SELEZIONE = 42
BASE = dict(thresh=0.95, confirm_ms=300, min_nom=0.10)
MODELLI = ["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"]
ETICHETTE = {"inc_fp32": "InceptionTime FP32", "inc_int8": "InceptionTime INT8",
             "ms_fp32": "MS-RCFE + Ridge FP32", "ms_int8": "MS-RCFE + Ridge INT8"}
LOG = []


def P(t=""):
    print(t)
    LOG.append(str(t))


# ------------------------------------------------------------------ etichettatura
def trova_t1(corrente, nominale, thresh, confirm_ms):
    """Come build_dataset_new.find_t1 (primo i con corrente < nominale*thresh per confirm campioni
    consecutivi, i < len - confirm), ma lineare nella lunghezza."""
    c = int(confirm_ms / 1000 * FS)
    sotto = corrente < nominale * thresh
    n = len(sotto)
    if n <= c:
        return None
    d = np.diff(np.concatenate(([0], sotto.astype(np.int8), [0])))
    inizi, fini = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(inizi, fini):
        if b - a >= c and a < n - c:
            return int(a)
    return None


def finestre_etichettate(n_tot, t1):
    """Stessa logica di extract_windows: restituisce (starts0, starts1)."""
    s0, s1 = [], []
    for start in range(0, n_tot - WIN_N + 1, STEP_N):
        end = start + WIN_N
        if t1 is None or end <= t1:
            s0.append(start)
        elif start >= t1:
            s1.append(start)
    return s0, s1


def seleziona(s0, s1):
    """Al massimo MAX_WIN finestre per classe, rng(42) nuovo per file, prima classe 0 poi 1."""
    rng = np.random.default_rng(SEED_SELEZIONE)
    out = []
    for lista in (s0, s1):
        if len(lista) > MAX_WIN:
            idx = rng.choice(len(lista), MAX_WIN, replace=False)
            lista = [lista[i] for i in sorted(idx)]
        out.append(lista)
    return out


# ------------------------------------------------------------------ dati
def indice_mat(root):
    """nome -> elenco dei percorsi (ordine di os.walk, come la v1 per la prima istanza)."""
    d = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".mat"):
                d.setdefault(fn, []).append(os.path.join(dp, fn))
    return d


def chiavi_istanze(files, mats):
    """Chiave per istanza: 'nome' per la prima, 'nome#2', 'nome#3' per le successive."""
    out = []
    for fn in files:
        for k, p in enumerate(mats[fn], 1):
            out.append((fn if k == 1 else f"{fn}#{k}", p))
    return out


def nome_base(chiave):
    return chiave.split("#")[0] if re.search(r"#\d+$", chiave) else chiave


def carica_correnti(istanze):
    from scipy.io import loadmat
    out = {}
    for i, (chiave, path) in enumerate(istanze, 1):
        print(f"\r  lettura .mat {i}/{len(istanze)}", end="", flush=True)
        m = loadmat(path, squeeze_me=False)
        out[chiave] = m["CurrentData"].flatten().astype(np.float64)
    print()
    return out


def calcola_score_finestre(repo, a, correnti, out_dir):
    """Score dei 4 modelli per TUTTE le finestre (passo 50 ms) di ogni istanza; cache incrementale."""
    cache = out_dir / "score_finestre.npz"
    chiavi = sorted(correnti)
    gia = {}
    if cache.exists():
        c = np.load(cache, allow_pickle=False)
        for i, fn in enumerate(c["files"]):
            gia[str(fn)] = c["score"][c["off"][i]:c["off"][i + 1]]
        P(f"Cache trovata: {len(gia)} istanze con score gia' calcolati")
    da_fare = [k for k in chiavi if k not in gia or len(gia[k]) != max(0, (len(correnti[k]) - WIN_N) // STEP_N + 1)]
    if da_fare:
        P(f"Inferenza su {len(da_fare)} istanze")
        sess = {k: a.apri_sessione(repo / a.MODELLI[k]) for k in MODELLI}
        coef = np.load(repo / a.RIDGE_COEF).astype(np.float64).reshape(-1)
        itc = float(np.load(repo / a.RIDGE_INTERCEPT).reshape(-1)[0])
        for i, fn in enumerate(da_fare, 1):
            cur = correnti[fn]
            nom = float(np.median(cur[:NOMINAL_N]))
            starts = np.arange(0, len(cur) - WIN_N + 1, STEP_N) if len(cur) >= WIN_N + STEP_N else np.array([], int)
            if nom < 0.05 or len(starts) == 0:
                gia[fn] = np.full((len(starts), 4), np.nan)
            else:
                X = np.stack([cur[s:s + WIN_N] / nom for s in starts]).astype(np.float32)
                sc = np.empty((len(starts), 4))
                for j, k in enumerate(MODELLI):
                    sc[:, j] = (a.score_inception(sess[k], X) if k.startswith("inc")
                                else a.score_msrcfe(sess[k], X, coef, itc))
                gia[fn] = sc
            if i % 25 == 0 or i == len(da_fare):
                P(f"  inferenza {i}/{len(da_fare)} ({time.strftime('%H:%M:%S')})")
        tutte = sorted(gia)
        off = np.concatenate(([0], np.cumsum([len(gia[k]) for k in tutte])))
        np.savez(cache, files=np.array(tutte), off=off, score=np.concatenate([gia[k] for k in tutte]))
    return {k: gia[k] for k in chiavi}


# ------------------------------------------------------------------ valutazione
def costruisci_variante(correnti, score_f, thresh, confirm_ms, min_nom):
    righe_y, righe_s, righe_f, t1s = [], [], [], {}
    for fn, cur in correnti.items():
        if len(cur) < WIN_N + STEP_N:
            continue
        nom = float(np.median(cur[:NOMINAL_N]))
        if nom < min_nom:
            continue
        t1 = trova_t1(cur, nom, thresh, confirm_ms)
        t1s[fn] = t1
        s0, s1 = seleziona(*finestre_etichettate(len(cur), t1))
        sc = score_f[fn]
        for lab, lista in ((0, s0), (1, s1)):
            for s in lista:
                righe_y.append(lab)
                righe_s.append(sc[s // STEP_N])
                righe_f.append(nome_base(fn))
    return np.array(righe_y), np.array(righe_s), np.array(righe_f), t1s


def metriche_ci(y, S, files, n_boot, seed):
    chiavi = np.array([re.match(r"^(.*?)_Study\d+", f).group(1) for f in files])
    esp = np.unique(chiavi)
    cod = np.searchsorted(esp, chiavi)
    n_esp = len(esp)
    W = np.random.default_rng(seed).multinomial(n_esp, np.full(n_esp, 1.0 / n_esp), size=n_boot).astype(float)
    ris = {}
    for j, k in enumerate(MODELLI):
        pred = (S[:, j] > 0).astype(int)
        c = np.zeros((n_esp, 4))
        for q, m in enumerate(((pred == 1) & (y == 1), (pred == 0) & (y == 1),
                               (pred == 0) & (y == 0), (pred == 1) & (y == 0))):
            c[:, q] = np.bincount(cod[m], minlength=n_esp)
        tp, fn_, tn, fp = c.sum(0)
        t = W @ c
        with np.errstate(divide="ignore", invalid="ignore"):
            dr = t[:, 0] / (t[:, 0] + t[:, 1])
            fpr = t[:, 3] / (t[:, 2] + t[:, 3])
            acc = (t[:, 0] + t[:, 2]) / t.sum(1)
        ci = lambda v: [float(x) * 100 for x in np.nanpercentile(v, [2.5, 97.5])]
        ris[k] = dict(DR=100 * tp / (tp + fn_) if tp + fn_ else float("nan"),
                      FPR=100 * fp / (tn + fp) if tn + fp else float("nan"),
                      ACC=100 * (tp + tn) / c.sum(), DR_ci=ci(dr), FPR_ci=ci(fpr), ACC_ci=ci(acc),
                      FN=int(fn_), FP=int(fp))
    return ris


def fmt(r, nome, d=1):
    v, (lo, hi) = r[nome], r[nome + "_ci"]
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ------------------------------------------------------------------ selftest
def selftest():
    ok = True
    x = np.ones(6000)
    x[1000:4500] = 0.5                                          # 3500 campioni (350 ms) sotto soglia
    ok &= trova_t1(x, 1.0, 0.95, 300) == 1000
    ok &= trova_t1(x, 1.0, 0.95, 500) is None                   # conferma di 500 ms non raggiunta
    ok &= trova_t1(x, 1.0, 0.40, 300) is None                   # soglia sotto il livello del segnale
    s0, s1 = finestre_etichettate(6000, 1000)
    ok &= all(s + WIN_N <= 1000 for s in s0) and all(s >= 1000 for s in s1)
    a0, a1 = seleziona(list(range(0, 100 * STEP_N, STEP_N)), list(range(0, 50 * STEP_N, STEP_N)))
    ok &= len(a0) == MAX_WIN and len(a1) == MAX_WIN
    b0, b1 = seleziona(list(range(0, 100 * STEP_N, STEP_N)), list(range(0, 50 * STEP_N, STEP_N)))
    ok &= a0 == b0 and a1 == b1                                  # deterministica
    # equivalenza con l'implementazione della pipeline (versione quadratica)
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = np.where(rng.random(3000) < 0.7, 0.5, 1.0)
        c = int(300 / 1000 * FS) // 10                           # confronto con c piccolo
        sotto = v < 0.95
        ref = next((i for i in range(len(sotto) - c) if sotto[i:i + c].all()), None)
        d = np.diff(np.concatenate(([0], sotto.astype(np.int8), [0])))
        got = next((int(a) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
                    if b - a >= c and a < len(sotto) - c), None)
        ok &= ref == got
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_sensibilita")
    ap.add_argument("--soglie", type=float, nargs="+", default=[0.90, 0.93, 0.97, 0.98])
    ap.add_argument("--conferme", type=float, nargs="+", default=[100, 200, 500, 1000], help="ms")
    ap.add_argument("--min-nom", type=float, nargs="+", default=[0.05, 0.2, 0.5], help="A")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-file", type=int, default=None, help="prova rapida: usa solo i primi N file (il controllo del caso base non e' valido)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.mat_root:
        sys.exit("serve --mat-root (cartella dei .mat a 10 kHz) oppure --selftest")

    qui = Path(__file__).resolve().parent
    repo = Path(args.repo) if args.repo else (qui.parent if (qui.parent / "dataset").is_dir() else Path.cwd())
    sys.path.insert(0, str(repo / "script"))
    import analisi_test_set_v1 as a
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    P(f"sensibilita_etichette {VERSIONE} - {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    files = sorted({m["filename"] for m in meta})
    if args.max_file:
        files = files[:args.max_file]
        meta = [m for m in meta if m["filename"] in set(files)]
    atteso = {}
    for m in meta:
        atteso[(m["filename"], int(m["label"]))] = atteso.get((m["filename"], int(m["label"])), 0) + 1
    mats = indice_mat(args.mat_root)
    mancanti = [x for x in files if x not in mats]
    if mancanti:
        sys.exit(f"{len(mancanti)} file .mat non trovati in {args.mat_root}, es. {mancanti[:3]}")
    istanze = chiavi_istanze(files, mats)
    omonimi = sum(len(mats[f]) - 1 for f in files)
    P(f"Istanze .mat: {len(istanze)} (di cui {omonimi} omonime in cartelle diverse)")
    P(f"File del test set: {len(files)}; esperimenti: {len({m['exp_key'] for m in meta})}")
    correnti = carica_correnti(istanze)
    score_f = calcola_score_finestre(repo, a, correnti, out)

    varianti = [("base", BASE["thresh"], BASE["confirm_ms"], BASE["min_nom"])]
    varianti += [(f"soglia {x:g}", x, BASE["confirm_ms"], BASE["min_nom"]) for x in args.soglie if x != BASE["thresh"]]
    varianti += [(f"conferma {x:g} ms", BASE["thresh"], x, BASE["min_nom"]) for x in args.conferme if x != BASE["confirm_ms"]]
    varianti += [(f"I min {x:g} A", BASE["thresh"], BASE["confirm_ms"], x) for x in args.min_nom if x != BASE["min_nom"]]

    righe, riepilogo, t1_base = [], {}, None
    for nome, th, cf, mn in varianti:
        y, S, fl, t1s = costruisci_variante(correnti, score_f, th, cf, mn)
        if nome == "base":
            t1_base = t1s
            cont = {}
            for fn, lab in zip(fl, y):
                cont[(fn, int(lab))] = cont.get((fn, int(lab)), 0) + 1
            coincide = cont == atteso
            P(f"\nControllo caso base: {len(y)} finestre ({int((y == 1).sum())} arco, {int((y == 0).sum())} pre-arco); "
              f"coincide con il test set pubblicato per ogni file: {'SI' if coincide else 'NO'}")
            if not coincide:
                P("  ATTENZIONE: il caso base non riproduce il test set (file .mat diversi o sottocampionati?). "
                  "I risultati delle varianti non sono confrontabili con la tesi.")
        sh = [(t1s[f] - t1_base[f]) / FS * 1000 for f in t1s if f in t1_base and t1s[f] is not None and t1_base[f] is not None]
        senza = sum(1 for f in t1s if t1s[f] is None)
        r = metriche_ci(y, S, fl, args.n_boot, args.seed)
        P(f"\n== {nome} ==  finestre {len(y)} (arco {int((y == 1).sum())}, pre-arco {int((y == 0).sum())}); "
          f"file senza arco {senza}; spostamento di t1 mediano {np.median(sh) if sh else float('nan'):+.0f} ms")
        P(f"{'Modello':<24}{'DR %':>26}{'FPR %':>26}{'Acc. %':>28}")
        for k in MODELLI:
            P(f"{ETICHETTE[k]:<24}{fmt(r[k], 'DR'):>26}{fmt(r[k], 'FPR'):>26}{fmt(r[k], 'ACC', 2):>28}")
            righe.append([nome, th, cf, mn, len(y), int((y == 1).sum()), int((y == 0).sum()), ETICHETTE[k],
                          f"{r[k]['DR']:.3f}", *[f"{v:.3f}" for v in r[k]["DR_ci"]], f"{r[k]['FPR']:.3f}",
                          *[f"{v:.3f}" for v in r[k]["FPR_ci"]], f"{r[k]['ACC']:.3f}",
                          *[f"{v:.3f}" for v in r[k]["ACC_ci"]], r[k]["FN"], r[k]["FP"]])
        riepilogo[nome] = dict(soglia=th, conferma_ms=cf, min_nom_A=mn, n_finestre=int(len(y)),
                               n_arco=int((y == 1).sum()), n_pre=int((y == 0).sum()), file_senza_arco=senza,
                               shift_t1_mediano_ms=(float(np.median(sh)) if sh else None),
                               modelli={ETICHETTE[k]: r[k] for k in MODELLI})
    with open(out / "sensibilita_etichette.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variante", "soglia", "conferma_ms", "min_nom_A", "n_finestre", "n_arco", "n_pre", "modello",
                    "DR", "DR_lo", "DR_hi", "FPR", "FPR_lo", "FPR_hi", "ACC", "ACC_lo", "ACC_hi", "FN", "FP"])
        w.writerows(righe)
    (out / "riepilogo_sensibilita.json").write_text(json.dumps(riepilogo, indent=1, ensure_ascii=False), encoding="utf-8")
    P("\nLimiti: i modelli sono addestrati con le etichette del caso base; una variante cambia le etichette e "
      "l'insieme delle finestre selezionate (al piu' 10 per classe e per file), non i modelli. Una sola esecuzione "
      "sul test set; gli intervalli non includono la variabilita' di addestramento.")
    (out / "report.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    P(f"\nFile scritti in: {out.resolve()}")


if __name__ == "__main__":
    main()
