#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indagine_ritardi_v1.py - perche' alcuni esperimenti hanno ritardi di rilevazione di 1-2 secondi
==============================================================================================

Per alcuni esperimenti ad alta corrente (3.5-5 A) i modelli INT8 rilevano l'arco con un ritardo di
1.4-2.0 s rispetto a t1 (Tab. 3.9). Questo script mostra, finestra per finestra, cosa succede in
quei file: livello della corrente (media della finestra / nominale), minimo e massimo normalizzati,
punteggio di ciascun modello e decisione. Serve a distinguere tre ipotesi:

  A) la corrente subito dopo t1 e' ancora vicina alla soglia 0.95 (livello ~0.9-1.0): il modello
     non vede una differenza netta e l'etichetta "arco" e' ambigua (limite delle etichette);
  B) il livello e' chiaramente basso (<0.8) ma il modello INT8 sbaglia: errore del modello
     (o della quantizzazione), da confrontare con la versione FP32;
  C) t1 e' anticipato rispetto all'arco reale (la corrente cala per altri motivi): limite di t1.

Per ogni file stampa le finestre da --prima ms prima di t1 fino a --dopo ms dopo la prima
rilevazione del modello piu' lento, e salva il CSV completo e un grafico (se matplotlib c'e').

Selezione dei file
    --file TESTO [TESTO ...]   parti del nome dei file (es. E004_0350V E007_0500V E006_0450V)
    oppure --csv ritardo_per_file.csv --top N   i N file con ritardo maggiore (K = 1)

Uso (dalla cartella della repo, come gli altri script)
    python -B script/indagine_ritardi_v1.py --repo . --mat-root "<cartella .mat>" --out ..\\ris_indagine ^
        --csv ..\\ris_ritardo\\ris_ritardo\\ritardo_per_file.csv --top 4
    python -B script/indagine_ritardi_v1.py --repo . --mat-root "<cartella .mat>" --out ..\\ris_indagine ^
        --file E004_0350V E007_0500V E006_0450V
    python -B script/indagine_ritardi_v1.py --repo . --selftest
Richiede norm_causale_v2.py e analisi_test_set_v1.py (o test_set_analysis.py rinominato) come
ritardo_rilevazione_v1.py. Non modifica nulla.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent
# La repo e' stata riordinata in sottocartelle: cerco i moduli in script/ e nelle sue sottocartelle.
for _d in [QUI, QUI.parent] + [x for r in (QUI, QUI.parent / "script") if r.is_dir() for x in r.iterdir() if x.is_dir()]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import norm_causale_v2 as nc  # noqa: E402

MODELLI = ["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"]


def livelli_finestre(i, nom, win, step):
    """Per ogni finestra: fine (campioni), media, minimo, massimo normalizzati sul nominale."""
    starts = np.arange(0, len(i) - win + 1, step)
    ends = starts + win
    liv = np.array([i[s:s + win].mean() for s in starts]) / nom
    mn = np.array([i[s:s + win].min() for s in starts]) / nom
    mx = np.array([i[s:s + win].max() for s in starts]) / nom
    return starts, ends, liv, mn, mx


def selezione_da_csv(path, top):
    righe = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    peggio = {}
    for r in righe:
        if r["K"] != "1" or r["ritardo_ms"] in ("", "None"):
            continue
        v = float(r["ritardo_ms"])
        peggio[r["file"]] = max(peggio.get(r["file"], 0.0), v)
    ordine = sorted(peggio, key=lambda f: -peggio[f])
    return ordine[:top], peggio


def selftest():
    fs, win, step = 10000, 1000, 500
    i = np.concatenate([np.full(20000, 4.0), np.full(20000, 2.0)])
    starts, ends, liv, mn, mx = livelli_finestre(i, 4.0, win, step)
    ok = len(starts) == (len(i) - win) // step + 1
    ok &= abs(liv[0] - 1.0) < 1e-12 and abs(liv[-1] - 0.5) < 1e-12
    ok &= (mn <= liv + 1e-12).all() and (mx >= liv - 1e-12).all()
    # finestra a cavallo di t1 = 20000: livello intermedio
    k = int(np.where(ends == 20500)[0][0])
    ok &= 0.5 < liv[k] < 1.0
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_indagine")
    ap.add_argument("--file", nargs="+", default=None, help="parti del nome dei file")
    ap.add_argument("--csv", default=None, help="ritardo_per_file.csv per scegliere i peggiori")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--prima", type=float, default=200.0, help="ms prima di t1 da mostrare")
    ap.add_argument("--dopo", type=float, default=300.0, help="ms dopo la rilevazione piu' lenta")
    ap.add_argument("--downsample-factor", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.mat_root:
        sys.exit("serve --mat-root oppure --selftest")

    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    p = nc.carica_parametri(repo)
    sys.path.insert(0, str(repo / "script"))
    try:
        import analisi_test_set_v1 as a
    except ImportError:          # nome nuovo dopo il riordino della repo
        import test_set_analysis as a
    from scipy.io import loadmat

    with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
        test_files = sorted({m["filename"] for m in csv.DictReader(f)})
    mats = nc.indice_mat(args.mat_root)

    if args.file:
        scelti = [f for f in test_files if any(t in f for t in args.file)]
    elif args.csv:
        scelti, _ = selezione_da_csv(args.csv, args.top)
    else:
        sys.exit("indica --file oppure --csv")
    scelti = [f for f in scelti if f in mats]
    if not scelti:
        sys.exit("nessun file selezionato (nome non trovato tra i .mat)")

    sess = {k: a.apri_sessione(repo / a.MODELLI[k]) for k in MODELLI}
    coef = np.load(repo / a.RIDGE_COEF).astype(np.float64).reshape(-1)
    itc = float(np.load(repo / a.RIDGE_INTERCEPT).reshape(-1)[0])
    fs, win, step = p.FS_HZ, p.WIN_N, p.STEP_N

    def punteggi(k, X):
        return a.score_inception(sess[k], X) if k.startswith("inc") else a.score_msrcfe(sess[k], X, coef, itc)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    riepilogo = []
    for fn in scelti:
        i = loadmat(mats[fn], squeeze_me=False)["CurrentData"].flatten()[::args.downsample_factor].astype(np.float64)
        t1, nom = p.find_t1(i)
        if t1 is None:
            print(f"\n{fn}: nessun arco rilevato, saltato")
            continue
        starts, ends, liv, mn, mx = livelli_finestre(i, nom, win, step)
        X = np.stack([i[s:s + win] / nom for s in starts]).astype(np.float32)
        sc = {k: punteggi(k, X) for k in MODELLI}
        dec = {k: sc[k] > 0 for k in MODELLI}
        t_ms = (ends - t1) / fs * 1000
        # prima decisione positiva dopo t1 (K = 1) per modello
        prima = {}
        for k in MODELLI:
            idx = np.where((ends > t1) & dec[k])[0]
            prima[k] = float(t_ms[idx[0]]) if len(idx) else None
        fine = max([v for v in prima.values() if v is not None] + [0.0]) + args.dopo
        sel = (t_ms >= -args.prima) & (t_ms <= fine)

        print(f"\n=== {fn}\n    nominale {nom:.3f} A, t1 = {t1 / fs:.3f} s; prima decisione positiva dopo t1 [ms]: "
              + ", ".join(f"{k}={('%.0f' % v) if v is not None else 'mai'}" for k, v in prima.items()))
        print(f"    {'fine-t1[ms]':>11} {'livello':>8} {'min':>6} {'max':>6}  "
              + "  ".join(f"{k:>9}" for k in MODELLI) + "   (punteggio, * = positivo)")
        for j in np.where(sel)[0]:
            celle = "  ".join(f"{sc[k][j]:>8.2f}{'*' if dec[k][j] else ' '}" for k in MODELLI)
            print(f"    {t_ms[j]:>11.0f} {liv[j]:>8.3f} {mn[j]:>6.2f} {mx[j]:>6.2f}  {celle}")

        # statistiche sul tratto tra t1 e la rilevazione dell'INT8 piu' lento
        lento = max((v for v in prima.values() if v is not None), default=None)
        tratto = (ends > t1) & (t_ms <= (lento if lento is not None else 0))
        riep = dict(file=fn, nominale_A=float(nom), t1_s=t1 / fs, prima_decisione_ms=prima,
                    n_finestre_nel_ritardo=int(tratto.sum()),
                    livello_medio_nel_ritardo=float(liv[tratto].mean()) if tratto.any() else None,
                    livello_min_nel_ritardo=float(liv[tratto].min()) if tratto.any() else None,
                    livello_max_nel_ritardo=float(liv[tratto].max()) if tratto.any() else None,
                    finestre_con_livello_sotto_095=int((liv[tratto] < 0.95).sum()) if tratto.any() else 0)
        riepilogo.append(riep)
        print(f"    -> nel ritardo: {riep['n_finestre_nel_ritardo']} finestre, livello medio "
              f"{riep['livello_medio_nel_ritardo']}, min {riep['livello_min_nel_ritardo']}, "
              f"max {riep['livello_max_nel_ritardo']}, di cui sotto 0.95: {riep['finestre_con_livello_sotto_095']}")

        nome = fn.replace(" ", "_").replace(".mat", "")
        with open(out / f"finestre_{nome}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fine_meno_t1_ms", "livello", "min", "max"] + [f"score_{k}" for k in MODELLI]
                       + [f"dec_{k}" for k in MODELLI])
            for j in range(len(ends)):
                w.writerow([round(float(t_ms[j]), 1), round(float(liv[j]), 5), round(float(mn[j]), 4),
                            round(float(mx[j]), 4)] + [round(float(sc[k][j]), 4) for k in MODELLI]
                           + [int(dec[k][j]) for k in MODELLI])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            m = sel
            ax[0].plot(t_ms[m], liv[m], "k.-", label="livello medio")
            ax[0].fill_between(t_ms[m], mn[m], mx[m], color="0.85", label="min-max")
            ax[0].axhline(0.95, color="r", ls="--", lw=0.8)
            ax[0].axvline(0, color="b", ls=":", lw=0.8)
            ax[0].set_ylabel("corrente / nominale")
            ax[0].legend(loc="best", fontsize=8)
            for k in MODELLI:
                ax[1].plot(t_ms[m], sc[k][m], ".-", label=k)
            ax[1].axhline(0, color="k", lw=0.5)
            ax[1].axvline(0, color="b", ls=":", lw=0.8)
            ax[1].set_xlabel("fine finestra - t1 [ms]")
            ax[1].set_ylabel("punteggio (>0 = arco)")
            ax[1].legend(loc="best", fontsize=8)
            fig.suptitle(fn, fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"grafico_{nome}.png", dpi=130)
            plt.close(fig)
        except Exception as e:  # il grafico e' facoltativo
            print(f"    (grafico non creato: {e})")

    with open(out / "riepilogo_indagine.json", "w", encoding="utf-8") as f:
        json.dump(dict(versione=VERSIONE, args={k: v for k, v in vars(args).items() if k != "mat_root"},
                       file=riepilogo), f, indent=1)
    print(f"\nSalvato in {out}/ (finestre_*.csv, grafico_*.png, riepilogo_indagine.json)")


if __name__ == "__main__":
    main()
