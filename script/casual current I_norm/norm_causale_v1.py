#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
norm_causale_v1.py - normalizzazione con I_nom causale (non statica), senza riaddestrare
========================================================================================

Il dataset pubblicato normalizza ogni registrazione con una corrente nominale statica
(mediana dei primi 0.2 s, ``build_dataset_new.find_t1``), che su un impianto reale non e'
disponibile. Questo script sostituisce quella stima con una stima CAUSALE: per ogni finestra
si usano solo campioni precedenti. Poi valuta i modelli gia' pubblicati sulle stesse finestre
del test set, rinormalizzate, senza riaddestrare.

Stimatore (per ogni finestra da WINDOW_S, passo STEP_S)
  1. valore della finestra v = mediana dei campioni di corrente (in ampere);
  2. I_nom(t) = mediana degli ultimi K = W / STEP_S valori accettati nel buffer, calcolata
     PRIMA di includere la finestra corrente;
  3. finestra normalizzata = corrente / I_nom(t);
  4. aggiornamento del buffer con la regola di gate (--gate):
       none   sempre (nessuna protezione: serve a mostrare che la stima segue l'arco)
       level  solo se v >= THRESH_FRAC * I_nom(t)   (usa il solo livello, nessun modello)
       model  solo se il classificatore NON sospetta un arco (anello chiuso con un modello)
  5. avvio: finche' il buffer e' vuoto e prima di NOMINAL_S secondi di storia, I_nom e' la
     mediana dei campioni disponibili fino alla fine della finestra corrente (avvio a freddo,
     dichiarato come limite). Dopo NOMINAL_S il buffer e' inizializzato con i valori di
     quei primi secondi.
  6. I_nom e' limitato inferiormente a MIN_NOM_A (come in build_dataset_new).

Le ETICHETTE restano quelle del dataset pubblicato (t1 da find_t1 con nominale statico):
il confronto misura la tolleranza dei modelli a un errore di scala, non una nuova
etichettatura.

Uso
    python norm_causale_v1.py --repo /percorso/DC-ARC-Fault --mat-root /dati/dc-arc-fault \
                              --out risultati_norm_causale --W 1 2 5 --gate level
    python norm_causale_v1.py --selftest            # prova su segnali sintetici, senza dati

--mat-root: cartella con i .mat estratti a 10 kHz (dataset Kaggle yassirsuarez/dc-arc-fault),
            cercati per nome file, ricorsivamente.
Dipendenze: numpy, scipy, onnxruntime, scikit-learn (come analisi_test_set_v1.py).
"""
import argparse
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# Parametri condivisi con la pipeline di costruzione del dataset
# --------------------------------------------------------------------------------------
def carica_parametri(repo):
    """Importa le costanti da build_dataset_new.py, cosi' restano identiche alla pipeline."""
    sys.path.insert(0, str(Path(repo) / "dataset"))
    import build_dataset_new as b  # noqa: E402
    return b


# --------------------------------------------------------------------------------------
# Stimatore causale
# --------------------------------------------------------------------------------------
def nominale_causale(corrente, W_s, gate, p, scorer=None):
    """Percorre TUTTE le finestre di una registrazione e restituisce, per ciascuna,
    (start, I_nom_causale, accettata_nel_buffer).

    corrente: array 1-D in ampere. p: modulo build_dataset_new (costanti).
    scorer(finestra_normalizzata) -> score; usato solo con gate == "model" (score > 0 = arco).
    """
    win_n, step_n, nom_n = p.WIN_N, p.STEP_N, p.NOMINAL_N
    K = max(1, int(round(W_s / p.STEP_S)))
    buffer = deque(maxlen=K)
    n = len(corrente)
    out = []
    seeded = False
    start = 0
    while start + win_n <= n:
        end = start + win_n
        v = float(np.median(corrente[start:end]))
        # --- stima I_nom con la sola storia disponibile -----------------------------
        if not seeded and end >= nom_n:
            # storia sufficiente: il buffer parte dai valori delle finestre gia' viste
            seeded = True
            s = 0
            while s + win_n <= start:           # finestre interamente prima di questa
                buffer.append(float(np.median(corrente[s:s + win_n])))
                s += step_n
        if len(buffer) == 0 or not seeded:
            i_nom = float(np.median(corrente[:end]))   # avvio a freddo
        else:
            i_nom = float(np.median(buffer))
        i_nom = max(i_nom, p.MIN_NOM_A)
        # --- gate di aggiornamento ---------------------------------------------------
        if gate == "none":
            accetta = True
        elif gate == "level":
            accetta = v >= p.THRESH_FRAC * i_nom
        elif gate == "model":
            x = (corrente[start:end] / i_nom).astype(np.float32)
            accetta = not (scorer(x) > 0)
        else:
            raise ValueError(gate)
        if seeded and accetta:
            buffer.append(v)
        out.append((start, i_nom, accetta))
        start += step_n
    return out


# --------------------------------------------------------------------------------------
# Self-test su segnali sintetici
# --------------------------------------------------------------------------------------
def selftest(p):
    rng = np.random.default_rng(0)
    fs = p.FS_HZ
    t = np.arange(int(4 * fs)) / fs
    i0 = 2.0
    stazionario = i0 + 0.01 * rng.standard_normal(len(t))
    # arco: la corrente scende a 1.2 A e diventa rumorosa dopo 1.5 s
    arco = stazionario.copy()
    arco[t >= 1.5] = 1.2 + 0.3 * rng.standard_normal((t >= 1.5).sum())
    ok = True

    def rapporto(sig, W, gate):
        r = nominale_causale(sig, W, gate, p)
        tt = np.array([s / fs for s, _, _ in r])
        return tt, np.array([i for _, i, _ in r]) / i0

    for W in (1, 2, 5):
        tt, r = rapporto(stazionario, W, "level")
        e = np.abs(r - 1).max()
        print(f"[stazionario] W={W} s  gate=level  max|I_nom/I0-1| = {100*e:.3f}%")
        ok &= e < 0.005
    tt, r_none = rapporto(arco, 2, "none")
    tt, r_lvl = rapporto(arco, 2, "level")
    fine = tt > 3.5
    print(f"[arco]        W=2 s  fine registrazione: gate=none  I_nom/I0 = {r_none[fine].mean():.3f}"
          f"   gate=level  I_nom/I0 = {r_lvl[fine].mean():.3f}")
    ok &= r_none[fine].mean() < 0.7          # senza gate la stima segue l'arco
    ok &= abs(r_lvl[fine].mean() - 1) < 0.01  # con gate resta congelata sul nominale
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# Valutazione sul test set
# --------------------------------------------------------------------------------------
def indice_mat(root):
    d = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".mat"):
                d.setdefault(fn, os.path.join(dp, fn))
    return d


def costruisci_scorer(repo, nome, a):
    sess = a.apri_sessione(repo / a.MODELLI[nome])
    inp = sess.get_inputs()[0].name
    if nome.startswith("inc"):
        def f(x):
            lg = sess.run(None, {inp: x.reshape(1, 1, -1)})[0].astype(np.float64)[0]
            return lg[1] - lg[0]
    else:
        coef = np.load(repo / a.RIDGE_COEF).astype(np.float64).reshape(-1)
        itc = float(np.load(repo / a.RIDGE_INTERCEPT).reshape(-1)[0])

        def f(x):
            ft = sess.run(None, {inp: x.reshape(1, 1, -1)})[0].astype(np.float64).reshape(-1)
            return float(ft @ coef + itc)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="risultati_norm_causale")
    ap.add_argument("--W", type=float, nargs="+", default=[2.0], help="ampiezze in secondi")
    ap.add_argument("--gate", choices=["none", "level", "model"], default="level")
    ap.add_argument("--gate-model", default="inc_fp32",
                    choices=["inc_fp32", "ms_fp32"], help="modello per --gate model")
    ap.add_argument("--downsample-factor", type=int, default=1,
                    help="1 se i .mat sono gia' a 10 kHz (dataset Kaggle); 25 se a 250 kHz")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    p = carica_parametri(repo)
    if args.selftest:
        sys.exit(selftest(p))
    if not args.mat_root:
        sys.exit("serve --mat-root (cartella dei .mat a 10 kHz) oppure --selftest")

    sys.path.insert(0, str(repo / "script"))
    import analisi_test_set_v1 as a
    from scipy.io import loadmat

    d = np.load(repo / a.TEST_NPZ)
    X_pub, y = d["X"], d["y"].astype(int)
    with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    assert len(meta) == len(y)
    mats = indice_mat(args.mat_root)
    files = sorted({m["filename"] for m in meta})
    mancanti = [f for f in files if f not in mats]
    if mancanti:
        sys.exit(f"{len(mancanti)} file .mat del test set non trovati in {args.mat_root}, "
                 f"es. {mancanti[:3]}")
    per_file = {}
    for i, m in enumerate(meta):
        per_file.setdefault(m["filename"], []).append(i)

    scorer = costruisci_scorer(repo, args.gate_model, a) if args.gate == "model" else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = {k: a.apri_sessione(repo / a.MODELLI[k]) for k in a.ORDINE}
    coef = np.load(repo / a.RIDGE_COEF).astype(np.float64).reshape(-1)
    itc = float(np.load(repo / a.RIDGE_INTERCEPT).reshape(-1)[0])

    def punteggi(X):
        sc = {}
        for k in a.ORDINE:
            sc[k] = (a.score_inception(sess[k], X) if k.startswith("inc")
                     else a.score_msrcfe(sess[k], X, coef, itc))
        return sc

    riepilogo = []
    for W in args.W:
        Xc = np.zeros_like(X_pub)
        err = np.full(len(y), np.nan)          # I_nom_causale / I_nom_statico - 1
        diff_rif = 0.0
        for fn in files:
            raw = loadmat(mats[fn], squeeze_me=False)["CurrentData"].flatten()[::args.downsample_factor]
            raw = raw.astype(np.float64)
            _, nom_stat = p.find_t1(raw)
            righe = per_file[fn]
            # controllo di allineamento con il dataset pubblicato
            rif = float(meta[righe[0]]["nominal_A"])
            if abs(nom_stat - rif) > 1e-3 * max(rif, 1e-9):
                sys.exit(f"{fn}: nominale ricalcolato {nom_stat:.4f} A != metadati {rif:.4f} A "
                         f"(controlla --downsample-factor)")
            stima = nominale_causale(raw, W, args.gate, p, scorer)
            by_start = {s: i_nom for s, i_nom, _ in stima}
            for r in righe:
                s = int(round(float(meta[r]["t_start_s"]) * p.FS_HZ))
                i_nom = by_start[s]
                Xc[r] = (raw[s:s + p.WIN_N] / i_nom).astype(np.float32)
                err[r] = i_nom / nom_stat - 1
                diff_rif = max(diff_rif,
                               float(np.abs(raw[s:s + p.WIN_N] / nom_stat - X_pub[r]).max()))
        print(f"W={W} s gate={args.gate}: ricostruzione finestre pubblicate, "
              f"max scarto = {diff_rif:.2e} (atteso ~1e-6)")
        sc_ref, sc_c = punteggi(X_pub), punteggi(Xc)
        ea = np.abs(err)
        print(f"  |errore I_nom| mediana {100*np.median(ea):.2f}%  p95 {100*np.percentile(ea,95):.2f}%  "
              f"entro ±0.5%: {100*(ea<=0.005).mean():.1f}%  entro ±1%: {100*(ea<=0.01).mean():.1f}%")
        print(f"  {'Modello':<26}{'':>10}{'Acc.':>8}{'DR':>8}{'FPR':>8}{'FN':>6}{'FP':>6}")
        for k in a.ORDINE:
            for nome, sc in (("statico", sc_ref), ("causale", sc_c)):
                m = a.metriche(y, (sc[k] > 0).astype(int))
                print(f"  {a.ETICHETTE[k]:<26}{nome:>10}{100*m['acc']:>8.2f}{100*m['dr']:>8.2f}"
                      f"{100*m['fpr']:>8.2f}{m['fn']:>6}{m['fp']:>6}")
                riepilogo.append(dict(W_s=W, gate=args.gate, modello=k, nominale=nome,
                                      acc=m["acc"], dr=m["dr"], fpr=m["fpr"], fn=m["fn"], fp=m["fp"],
                                      err_mediano=float(np.median(ea)),
                                      err_p95=float(np.percentile(ea, 95)),
                                      entro_0_5=float((ea <= 0.005).mean())))
        np.savez_compressed(out / f"finestre_causali_W{W:g}_{args.gate}.npz",
                            X=Xc, y=y, err_nominale=err)
    with open(out / "riepilogo.json", "w", encoding="utf-8") as f:
        json.dump(dict(versione=VERSIONE, args=vars(args), risultati=riepilogo), f, indent=1)
    print(f"Salvato in {out}/")


if __name__ == "__main__":
    main()
