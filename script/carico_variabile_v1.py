#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carico_variabile_v2.py - la stima causale di I_nom regge una variazione di carico?
==================================================================================

Le registrazioni del dataset hanno carico stabile prima dell'arco. Su un impianto reale la
corrente cambia con l'irraggiamento. Questo script applica alla corrente di ogni registrazione
un guadagno g(t) SINTETICO (rampa o gradino, cioe' una variazione di carico che NON e' un arco) e
confronta due modi di normalizzare le stesse finestre del test set:

  statico   finestra / nominale statico del dataset (primi 0.2 s della registrazione ORIGINALE):
            non si adatta alla variazione;
  causale   finestra / I_nom stimato con norm_causale_v2.nominale_causale (stesso gate).

I modelli non vengono riaddestrati. Le etichette restano quelle del dataset (t1 della
registrazione originale): il guadagno g(t) e' applicato a tutta la traccia, pre-arco e arco.

NOVITA' v2 (--pad, default 4 s): le registrazioni durano ~4 s e l'arco inizia dopo 0.2-0.5 s, troppo presto
perche' la stima (mediana di ~2 s) possa seguire una variazione: la v1 non poteva mostrare l'adattamento.
La v2 antepone alla traccia --pad secondi di corrente a regime (i primi 0.2 s della registrazione ripetuti,
cioe' il funzionamento normale), applica il guadagno g(t) da --t0 (default 0.5 s) e poi lascia seguire la
registrazione originale con il suo arco. Le finestre del test set e le etichette sono le stesse (spostate
di --pad), quindi si valutano TUTTE le finestre pre-arco (2641) e di arco (4520). Gate chiuso ed errore di
I_nom sono misurati solo sulle finestre pre-arco (sulle finestre di arco il gate e' chiuso per costruzione).
Artefatto da tenere presente: la ripetizione dei primi 0.2 s crea piccoli salti ogni 0.2 s.

Scenari (--scenari), ognuno "tipo:valore":
  ramp:-0.01   rampa lineare a -1% al secondo a partire da t0 (g >= --gmin)
  ramp:+0.02   rampa a +2% al secondo
  step:-0.05   gradino istantaneo a -5% a t0
  step:+0.05   gradino a +5%
Il segno e' quello della variazione di corrente: negativo = carico che scende.

Misure per scenario e metodo:
  FPR su pre-arco   falsi allarmi dovuti alla sola variazione di carico (finestre label 0)
  DR su arco        rilevazioni (finestre label 1)
  % finestre con gate chiuso   quota di finestre in cui la stima non si e' aggiornata
  errore I_nom      I_nom causale / (nominale statico * g(t)) - 1  (mediana, 95 percentile |.|)

Uso
    python carico_variabile_v2.py --repo . --mat-root "<cartella dei .mat>" --out ris_carico \
        --scenari ramp:-0.01 ramp:-0.02 ramp:-0.05 step:-0.05 step:+0.05 --gate-frac 0.98
    python carico_variabile_v2.py --selftest

--modelli: default inc_fp32 ms_fp32 (piu' veloce); usa anche inc_int8 ms_int8 per i quattro.
Richiede norm_causale_v2.py nella stessa cartella.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

VERSIONE = "v2"
QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import norm_causale_v2 as nc  # noqa: E402


# --------------------------------------------------------------------------------------
# Guadagno sintetico
# --------------------------------------------------------------------------------------
def guadagno(t, tipo, valore, t0, gmin=0.5, gmax=2.0):
    """g(t) moltiplicativo. tipo 'ramp': valore = variazione relativa al secondo (con segno);
    tipo 'step': valore = variazione relativa istantanea a t0."""
    g = np.ones_like(t, dtype=np.float64)
    dopo = t >= t0
    if tipo == "ramp":
        g[dopo] = 1.0 + valore * (t[dopo] - t0)
    elif tipo == "step":
        g[dopo] = 1.0 + valore
    else:
        raise ValueError(tipo)
    return np.clip(g, gmin, gmax)


def parse_scenario(s):
    tipo, val = s.split(":")
    return tipo, float(val)


# --------------------------------------------------------------------------------------
# Self-test su segnali sintetici
# --------------------------------------------------------------------------------------
def selftest(p):
    fs = p.FS_HZ
    t = np.arange(int(6 * fs)) / fs
    rng = np.random.default_rng(1)
    base = 2.0 + 0.01 * rng.standard_normal(len(t))
    ok = True
    print("variazione di carico senza arco, gate 0.98, W = 2 s (I_nom/atteso a fine traccia)")
    for tipo, val in (("ramp", -0.01), ("ramp", -0.02), ("ramp", -0.05), ("step", -0.05), ("step", 0.05)):
        g = guadagno(t, tipo, val, 0.25)
        sig = base * g
        r = nc.nominale_causale(sig, 2.0, "level", p, None, 0.98)
        starts = np.array([s for s, _, _ in r])
        inom = np.array([i for _, i, _ in r])
        acc = np.array([a for _, _, a in r])
        atteso = 2.0 * g[starts + p.WIN_N // 2]
        e = (inom / atteso - 1)[starts / fs > 5.0]
        chiuse = 100 * (1 - acc[starts / fs >= 0.25].mean())
        print(f"  {tipo}{val:+.2f}: errore finale {100*e.mean():+6.2f}%   gate chiuso {chiuse:5.1f}% delle finestre")
        if (tipo, val) == ("ramp", -0.01):
            ok &= abs(e.mean()) < 0.03          # una rampa lenta e' seguita
        if (tipo, val) == ("step", -0.05):
            ok &= e.mean() > 0.04               # un gradino verso il basso blocca la stima
        if (tipo, val) == ("step", 0.05):
            ok &= abs(e.mean()) < 0.01          # un gradino verso l'alto e' seguito
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# Valutazione
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_carico")
    ap.add_argument("--scenari", nargs="+",
                    default=["ramp:-0.01", "ramp:-0.02", "ramp:-0.05", "step:-0.05", "step:+0.05"])
    ap.add_argument("--W", type=float, default=2.0)
    ap.add_argument("--gate", choices=["none", "level", "hyst"], default="level")
    ap.add_argument("--gate-frac", type=float, default=0.98)
    ap.add_argument("--pad", type=float, default=4.0, help="secondi di corrente a regime anteposti (0 = come v1)")
    ap.add_argument("--t0", type=float, default=None, help="inizio della variazione [s]; default 0.5 con --pad, 0.25 senza")
    ap.add_argument("--gmin", type=float, default=0.5)
    ap.add_argument("--modelli", nargs="+", default=["inc_fp32", "ms_fp32"],
                    choices=["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"])
    ap.add_argument("--downsample-factor", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    p = nc.carica_parametri(repo)
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
    t_start = np.array([float(m["t_start_s"]) for m in meta])
    mats = nc.indice_mat(args.mat_root)
    files = sorted({m["filename"] for m in meta})
    mancanti = [f for f in files if f not in mats]
    if mancanti:
        sys.exit(f"{len(mancanti)} file .mat non trovati in {args.mat_root}, es. {mancanti[:3]}")
    per_file = {}
    for i, m in enumerate(meta):
        per_file.setdefault(m["filename"], []).append(i)

    sess = {k: a.apri_sessione(repo / a.MODELLI[k]) for k in args.modelli}
    coef = np.load(repo / a.RIDGE_COEF).astype(np.float64).reshape(-1)
    itc = float(np.load(repo / a.RIDGE_INTERCEPT).reshape(-1)[0])

    def punteggi(X):
        return {k: (a.score_inception(sess[k], X) if k.startswith("inc")
                    else a.score_msrcfe(sess[k], X, coef, itc)) for k in args.modelli}

    t0 = args.t0 if args.t0 is not None else (0.5 if args.pad > 0 else 0.25)
    pad_n = int(round(args.pad * p.FS_HZ / p.STEP_N)) * p.STEP_N
    toccate = np.ones(len(y), bool) if pad_n > 0 else (t_start >= t0)
    pre, arco = toccate & (y == 0), toccate & (y == 1)
    print(f"pad {pad_n / p.FS_HZ:g} s, variazione da t0 = {t0:g} s: "
          f"{int(pre.sum())} finestre pre-arco, {int(arco.sum())} di arco valutate")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    risultati = []

    for scen in args.scenari:
        tipo, val = parse_scenario(scen)
        Xs = np.zeros_like(X_pub)
        Xc = np.zeros_like(X_pub)
        err = np.full(len(y), np.nan)
        chiuso = np.zeros(len(y), bool)
        for fn in files:
            raw = loadmat(mats[fn], squeeze_me=False)["CurrentData"].flatten()[::args.downsample_factor]
            raw = raw.astype(np.float64)
            _, nom_stat = p.find_t1(raw)
            rif = float(meta[per_file[fn][0]]["nominal_A"])
            if abs(nom_stat - rif) > 1e-3 * max(rif, 1e-9):
                sys.exit(f"{fn}: nominale ricalcolato {nom_stat:.4f} A != metadati {rif:.4f} A")
            if pad_n > 0:
                reps = int(np.ceil(pad_n / p.NOMINAL_N))
                sig = np.concatenate([np.tile(raw[:p.NOMINAL_N], reps)[:pad_n], raw])
            else:
                sig = raw
            t = np.arange(len(sig)) / p.FS_HZ
            g = guadagno(t, tipo, val, t0, args.gmin)
            mod = sig * g
            stima = nc.nominale_causale(mod, args.W, args.gate, p, None, args.gate_frac)
            by = {s: (i, acc) for s, i, acc in stima}
            for r in per_file[fn]:
                s = int(round(t_start[r] * p.FS_HZ)) + pad_n
                i_nom, acc = by[s]
                w = mod[s:s + p.WIN_N]
                Xs[r] = (w / nom_stat).astype(np.float32)
                Xc[r] = (w / i_nom).astype(np.float32)
                atteso = nom_stat * float(np.mean(g[s:s + p.WIN_N]))
                err[r] = i_nom / atteso - 1
                chiuso[r] = not acc
        sc_s, sc_c = punteggi(Xs), punteggi(Xc)
        ea = np.abs(err[pre])
        print(f"\n=== {scen}  (W = {args.W:g} s, gate {args.gate} {args.gate_frac}, pad {args.pad:g} s)")
        print(f"  sulle finestre pre-arco: gate chiuso {100*chiuso[pre].mean():.1f}%; "
              f"errore I_nom mediano {100*np.median(err[pre]):+.2f}%  95° perc. |.| {100*np.percentile(ea,95):.2f}%")
        print(f"  {'Modello':<26}{'metodo':>9}{'FPR pre-arco':>14}{'DR arco':>10}{'FP':>6}{'FN':>6}")
        for k in args.modelli:
            for nome, sc in (("statico", sc_s), ("causale", sc_c)):
                pred = (sc[k] > 0).astype(int)
                fp = int(((pred == 1) & pre).sum())
                fn_ = int(((pred == 0) & arco).sum())
                fpr = fp / max(int(pre.sum()), 1)
                dr = 1 - fn_ / max(int(arco.sum()), 1)
                print(f"  {a.ETICHETTE[k]:<26}{nome:>9}{100*fpr:>13.1f}%{100*dr:>9.1f}%{fp:>6}{fn_:>6}")
                risultati.append(dict(scenario=scen, modello=k, metodo=nome, fpr_pre=fpr, dr_arco=dr,
                                      fp=fp, fn=fn_, n_pre=int(pre.sum()), n_arco=int(arco.sum()),
                                      gate_chiuso=float(chiuso[pre].mean()),
                                      err_mediano=float(np.median(err[pre])),
                                      err_p95=float(np.percentile(ea, 95))))
        # un riepilogo per scenario: non viene mai sovrascritto da un altro scenario
        nome_file = scen.replace(":", "_").replace("+", "p").replace("-", "m")
        with open(out / f"riepilogo_{nome_file}.json", "w", encoding="utf-8") as f:
            json.dump(dict(versione=VERSIONE,
                           args={k: v for k, v in vars(args).items() if k != "mat_root"},
                           risultati=[r for r in risultati if r["scenario"] == scen]), f, indent=1)
    print(f"\nSalvato in {out}/ (un riepilogo_<scenario>.json per scenario; senza il percorso dei .mat)")


if __name__ == "__main__":
    main()