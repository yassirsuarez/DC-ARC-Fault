#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnosi_tensione_v1.py - la tensione (VoltageData) dei file .mat varia davvero?
================================================================================

Nel risultato dell'energia (energia_per_file.csv) la tensione media dopo t1 vale esattamente
50.0 V in tutti i 442 esperimenti, anche per ricette con tensione impostata diversa (150-500 V).
Questo fa pensare che VoltageData sia un canale costante (o saturato) e non una misura dell'arco.
Lo script lo verifica sui .mat e controlla anche Power (potenza) e il rapporto Power / (V * I).

Per ogni file del test set (o tutti con --tutti) scrive diagnosi_tensione.csv con:
    min, max, media, deviazione standard di VoltageData; numero di valori distinti;
    min, max, media di Power (se presente); media di Power / (V*I) dopo t1.
e stampa un riepilogo: in quanti file la tensione e' costante (deviazione standard < --eps V).

Uso (dalla cartella della repo)
    python -B script/diagnosi_tensione_v1.py --repo . --mat-root "<cartella dei .mat>" --out ..\\ris_diagnosi
    python -B script/diagnosi_tensione_v1.py --repo . --selftest
Non modifica nulla. Cerca i moduli in script/ e nelle sue sottocartelle.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent
_radici = [QUI] + ([QUI.parent] if QUI.parent.name == "script" else [])
for _r in list(_radici):
    _radici += [x for x in _r.iterdir() if x.is_dir()]
for _d in _radici:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import norm_causale_v2 as nc  # noqa: E402


def statistiche(v, eps):
    v = np.asarray(v, dtype=np.float64).ravel()
    return dict(v_min=float(v.min()), v_max=float(v.max()), v_media=float(v.mean()),
                v_std=float(v.std()), v_distinti=int(len(np.unique(np.round(v, 6)))),
                costante=bool(v.std() < eps))


def selftest():
    ok = statistiche(np.full(1000, 50.0), 1e-3)["costante"]
    ok &= not statistiche(np.r_[np.zeros(500), np.full(500, 50.0)], 1e-3)["costante"]
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_diagnosi")
    ap.add_argument("--eps", type=float, default=1e-3, help="deviazione standard sotto cui la tensione e' costante [V]")
    ap.add_argument("--tutti", action="store_true", help="tutti i .mat invece dei soli file del test set")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.mat_root:
        sys.exit("serve --mat-root oppure --selftest")

    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    p = nc.carica_parametri(repo)
    from scipy.io import loadmat
    mats = nc.indice_mat(args.mat_root)
    if args.tutti:
        files = sorted(mats)
    else:
        sys.path.insert(0, str(repo / "script"))
        try:
            import analisi_test_set_v1 as a
        except ImportError:
            import test_set_analysis as a
        with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
            files = sorted({m["filename"] for m in csv.DictReader(f)})
    righe, senza = [], []
    for fn in files:
        if fn not in mats:
            continue
        m = loadmat(mats[fn], squeeze_me=False)
        chiavi = {k.lower(): k for k in m if not k.startswith("__")}
        if "voltagedata" not in chiavi or "currentdata" not in chiavi:
            senza.append(fn)
            continue
        v = m[chiavi["voltagedata"]].flatten().astype(np.float64)
        i = m[chiavi["currentdata"]].flatten().astype(np.float64)
        n = min(len(v), len(i))
        v, i = v[:n], i[:n]
        r = dict(file=fn, **statistiche(v, args.eps))
        t1, _ = p.find_t1(i[::1])
        r["t1_s"] = t1 / p.FS_HZ if t1 is not None else ""
        if "power" in chiavi:
            w = m[chiavi["power"]].flatten().astype(np.float64)[:n]
            r.update(p_min=float(w.min()), p_max=float(w.max()), p_media=float(w.mean()))
            sl = slice(t1, None) if t1 is not None else slice(0, None)
            d = v[sl] * i[sl]
            r["rapporto_P_su_VI"] = float(np.mean(w[sl]) / np.mean(d)) if np.mean(d) != 0 else ""
        righe.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    campi = sorted({k for r in righe for k in r}, key=lambda k: (k != "file", k))
    with open(out / "diagnosi_tensione.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campi)
        w.writeheader()
        w.writerows(righe)
    n = len(righe)
    cost = sum(r["costante"] for r in righe)
    print(f"\nFile esaminati: {n} (senza Voltage/Current: {len(senza)})")
    print(f"Tensione costante (std < {args.eps:g} V): {cost} file su {n}")
    if n:
        vm = np.array([r["v_media"] for r in righe])
        vs = np.array([r["v_std"] for r in righe])
        print(f"Media di VoltageData per file: min {vm.min():.4f}  max {vm.max():.4f} V; "
              f"deviazione standard per file: min {vs.min():.4g}  max {vs.max():.4g} V")
        print(f"Valori distinti per file (mediana): {int(np.median([r['v_distinti'] for r in righe]))}")
        pw = [r for r in righe if "p_media" in r]
        if pw:
            print(f"Power presente in {len(pw)} file; Power / (V*I) mediano: "
                  f"{np.median([r['rapporto_P_su_VI'] for r in pw if r['rapporto_P_su_VI'] != '']):.4f}")
    print(f"\nSalvato in {out}/diagnosi_tensione.csv")


if __name__ == "__main__":
    main()
