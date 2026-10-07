#!/usr/bin/env python3
"""
mat_tools.py
============

Strumenti di pre-elaborazione per il dataset *Photovoltaic DC Arc Library*
[IEEE DataPort, https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library].

Il dataset originale, campionato a 250 kHz e corredato di file accessori non
informativi (immagini, README, fogli di calcolo), occupa diverse centinaia
di GB. Questo modulo fornisce due operazioni, eseguite una tantum prima
dell'addestramento, per ridurlo a una versione compatta e condivisibile:

``clean``
    Isola i soli file ``.mat`` dall'archivio grezzo, scartando metadati e
    file non pertinenti, replicando la gerarchia di cartelle originale.

``extract``
    Per ogni file ``.mat``, estrae le tre variabili fisiche rilevanti
    (corrente, tensione, potenza istantanea), applica un sottocampionamento
    e salva il risultato in formato compresso, verificando che i dati
    riletti coincidano con quelli scritti.

Il dataset ridotto prodotto da ``extract`` è quello pubblicato su Kaggle
(https://www.kaggle.com/datasets/yassirsuarez/dc-arc-fault) ed è il punto
di partenza della pipeline di addestramento (si veda ``build_dataset_new.py``).

Esempi d'uso
-------------
    python mat_tools.py clean  /dati/raw_dataset
    python mat_tools.py extract /dati/raw_dataset_clean
    python mat_tools.py extract /dati/raw_dataset_clean --downsample-factor 25
    python mat_tools.py extract singolo_studio.mat --window 2.0

Autori
------
Lorenzo Meloccaro, Yassir Flavio Suarez Sanchez
Corso di Laurea Magistrale in Ingegneria Informatica e dell'Automazione,
Universita' Politecnica delle Marche.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

__version__ = "2.0.0"

# ════════════════════════════════════════════════════════════════════════
# Parametri di default
# ════════════════════════════════════════════════════════════════════════

#: Nomi delle variabili da estrarre da ciascun file .mat. Il confronto con
#: le chiavi effettivamente presenti nel file e' case-insensitive, per
#: tollerare le differenze di capitalizzazione osservate nel dataset
#: originale (es. "power" vs "Power").
VARIABLES_TO_EXTRACT: tuple[str, ...] = ("CurrentData", "VoltageData", "power")

#: Frequenza di campionamento del dataset originale [Hz].
FS_ORIGINAL_HZ: int = 250_000

#: Frequenza di campionamento target dopo il sottocampionamento [Hz].
FS_TARGET_HZ: int = 10_000

#: Fattore di sottocampionamento derivato (1 campione ogni N viene tenuto).
DEFAULT_DOWNSAMPLE_FACTOR: int = FS_ORIGINAL_HZ // FS_TARGET_HZ  # = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("mat_tools")


# ════════════════════════════════════════════════════════════════════════
# Utilita'
# ════════════════════════════════════════════════════════════════════════

def _format_size(num_bytes: float) -> str:
    """Formatta una dimensione in byte in una stringa leggibile (es. ``'12.3 MB'``)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _mirror_path(input_root: Path, output_root: Path, filepath: Path) -> Path:
    """Restituisce il percorso di ``filepath`` rilocato sotto ``output_root``,
    preservando la struttura relativa rispetto a ``input_root``.
    """
    return output_root / filepath.relative_to(input_root)


def _find_mat_files(root: Path) -> list[Path]:
    """Cerca ricorsivamente tutti i file ``.mat`` sotto ``root``, in ordine
    deterministico (necessario per la riproducibilita' dei log e dei report).
    """
    return sorted(p for p in root.rglob("*") if p.suffix.lower() == ".mat")


# ════════════════════════════════════════════════════════════════════════
# Risultati delle operazioni (per report strutturati e riproducibilita')
# ════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ExtractionResult:
    """Esito dell'estrazione di un singolo file ``.mat``."""

    source: str
    destination: str | None
    status: str  # "ok" | "warning" | "error"
    message: str
    variables_found: list[str] = dataclasses.field(default_factory=list)
    variables_missing: list[str] = dataclasses.field(default_factory=list)
    original_size_bytes: int | None = None
    output_size_bytes: int | None = None


@dataclasses.dataclass
class BatchSummary:
    """Riepilogo di un'esecuzione su un intero albero di cartelle."""

    command: str
    input_root: str
    output_root: str
    downsample_factor: int | None
    n_total: int
    n_ok: int
    n_warning: int
    n_error: int
    results: list[ExtractionResult]

    def to_json(self, path: Path) -> None:
        """Salva il riepilogo in JSON, per tracciabilita' e riproducibilita'."""
        payload = dataclasses.asdict(self)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════
# clean — isola i soli file .mat
# ════════════════════════════════════════════════════════════════════════

def clean_directory(root_path: Path, *, dry_run: bool = False) -> BatchSummary:
    """Copia i soli file ``.mat`` di ``root_path`` in ``<root_path>_clean/``,
    preservando la struttura delle cartelle e scartando ogni altro file
    (immagini, metadati testuali, fogli di calcolo).

    Parameters
    ----------
    root_path:
        Cartella radice del dataset grezzo. Deve esistere.
    dry_run:
        Se ``True``, registra le operazioni che verrebbero eseguite senza
        copiare alcun file.

    Returns
    -------
    BatchSummary
        Riepilogo dell'operazione, con lo stato di ogni file trovato.

    Raises
    ------
    NotADirectoryError
        Se ``root_path`` non esiste o non e' una cartella.
    """
    if not root_path.is_dir():
        raise NotADirectoryError(f"cartella non trovata: {root_path}")

    output_root = root_path.parent / f"{root_path.name}_clean"
    log.info("output: %s", output_root)

    mat_files = _find_mat_files(root_path)
    if not mat_files:
        log.error("nessun file .mat trovato in: %s", root_path)
        return BatchSummary("clean", str(root_path), str(output_root), None, 0, 0, 0, 0, [])

    results: list[ExtractionResult] = []
    for src in mat_files:
        dst = _mirror_path(root_path, output_root, src)

        if dry_run:
            log.info("would-copy  %s  ->  %s", src, dst)
            results.append(ExtractionResult(str(src), str(dst), "ok", "dry-run"))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            log.info("copied      %s  ->  %s", src, dst)
            results.append(ExtractionResult(str(src), str(dst), "ok", "copiato"))
        except OSError as exc:
            log.error("failed      %s  (%s)", src, exc)
            results.append(ExtractionResult(str(src), None, "error", str(exc)))

    n_ok = sum(r.status == "ok" for r in results)
    n_error = sum(r.status == "error" for r in results)
    log.info("completato  totale=%d  ok=%d  errori=%d", len(results), n_ok, n_error)

    return BatchSummary("clean", str(root_path), str(output_root), None,
                         len(results), n_ok, 0, n_error, results)


# ════════════════════════════════════════════════════════════════════════
# extract — estrae le variabili fisiche e sottocampiona
# ════════════════════════════════════════════════════════════════════════

def _extract_single_file(
    mat_path: Path,
    input_root: Path,
    output_root: Path,
    *,
    downsample_factor: int,
    window_seconds: float | None,
) -> ExtractionResult:
    """Estrae le variabili di interesse da un singolo file ``.mat``,
    le sottocampiona e salva il risultato, verificando che i dati
    riletti dal file appena scritto coincidano con quelli estratti.
    """
    try:
        mat_full = loadmat(mat_path, squeeze_me=False)
    except NotImplementedError:
        msg = "formato HDF5/v7.3 non supportato da scipy.io.loadmat (richiede h5py)"
        log.error("%s: %s", mat_path, msg)
        return ExtractionResult(str(mat_path), None, "error", msg)
    except Exception as exc:  # noqa: BLE001 — vogliamo continuare il batch
        log.error("impossibile aprire %s: %s", mat_path, exc)
        return ExtractionResult(str(mat_path), None, "error", str(exc))

    available = {key for key in mat_full if not key.startswith("__")}
    log.info("file  %s  variabili_disponibili=%s", mat_path, sorted(available))

    extracted: dict[str, np.ndarray] = {}
    found: list[str] = []
    missing: list[str] = []

    for variable in VARIABLES_TO_EXTRACT:
        match = next((k for k in available if k.lower() == variable.lower()), None)
        if match is None:
            missing.append(variable)
            log.warning("variabile assente  %-12s  disponibili=%s", variable, sorted(available))
            continue
        if match != variable:
            log.warning("nome diverso da atteso  richiesta=%-12s  trovata=%s", variable, match)

        data = mat_full[match].flatten()[::downsample_factor]
        if window_seconds is not None:
            data = data[: int(FS_TARGET_HZ * window_seconds)]

        log.info(
            "estratta  %-12s  shape=%-14s  min=%.6f  max=%.6f  mean=%.6f  (fattore=%d)",
            match, str(data.shape), data.min(), data.max(), data.mean(), downsample_factor,
        )
        extracted[match] = data
        found.append(match)

    if not extracted:
        msg = "nessuna variabile estratta"
        log.error("%s: %s", mat_path, msg)
        return ExtractionResult(str(mat_path), None, "error", msg, found, missing)

    destination = _mirror_path(input_root, output_root, mat_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    savemat(destination, extracted, do_compression=True)

    original_size = mat_path.stat().st_size
    output_size = destination.stat().st_size
    saved_pct = (1 - output_size / original_size) * 100
    log.info(
        "salvato  %s  (%s -> %s, %.1f%% risparmiato)",
        destination, _format_size(original_size), _format_size(output_size), saved_pct,
    )

    # Verifica di integrita': rileggiamo il file appena scritto e lo
    # confrontiamo con i dati estratti in memoria, per garantire che la
    # compressione non abbia introdotto perdita o corruzione.
    reloaded = loadmat(destination, squeeze_me=False)
    mismatches = []
    for variable, original_values in extracted.items():
        # MATLAB non ha un vero array 1-D: scipy.io.savemat salva un
        # vettore come riga 2-D (1, N), che loadmat(squeeze_me=False)
        # restituisce cosi' com'e'. Appiattiamo prima del confronto,
        # altrimenti np.array_equal fallisce per sola differenza di
        # forma anche quando i valori coincidono esattamente.
        reloaded_values = reloaded[variable].flatten()
        if np.array_equal(original_values, reloaded_values):
            log.info("verifica  %-12s  OK", variable)
        else:
            max_diff = np.max(np.abs(original_values.astype(float) - reloaded_values.astype(float)))
            log.warning("verifica  %-12s  differenza massima=%.2e", variable, max_diff)
            mismatches.append(variable)

    status = "ok" if not mismatches and not missing else "warning"
    message = "estrazione completata" if status == "ok" else (
        f"variabili mancanti: {missing}" if missing else f"verifica fallita per: {mismatches}"
    )

    return ExtractionResult(
        source=str(mat_path),
        destination=str(destination),
        status=status,
        message=message,
        variables_found=found,
        variables_missing=missing,
        original_size_bytes=original_size,
        output_size_bytes=output_size,
    )


def extract_path(
    path: Path,
    *,
    downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    window_seconds: float | None = None,
) -> BatchSummary:
    """Estrae corrente, tensione e potenza da un file ``.mat`` o da ogni
    file ``.mat`` trovato ricorsivamente sotto una cartella, sottocampiona
    e salva il risultato in ``<cartella>_extracted/``.

    Parameters
    ----------
    path:
        File ``.mat`` singolo oppure cartella da scansionare ricorsivamente.
    downsample_factor:
        Tiene 1 campione ogni ``downsample_factor``. Il default riduce il
        dataset da 250 kHz a 10 kHz.
    window_seconds:
        Se specificato, tronca ogni segnale ai primi ``window_seconds``
        secondi dopo il sottocampionamento.

    Returns
    -------
    BatchSummary
        Riepilogo dell'operazione, con lo stato di ogni file processato.

    Raises
    ------
    ValueError
        Se ``downsample_factor`` non e' un intero positivo, o se ``path``
        non e' ne' un file ``.mat`` ne' una cartella esistente.
    """
    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor deve essere positivo, ricevuto {downsample_factor}")

    if path.is_file():
        if path.suffix.lower() != ".mat":
            raise ValueError(f"estensione non valida (atteso .mat): {path}")
        input_root = path.parent
        output_root = input_root.parent / f"{input_root.name}_extracted"
        log.info("output: %s", output_root)
        result = _extract_single_file(
            path, input_root, output_root,
            downsample_factor=downsample_factor, window_seconds=window_seconds,
        )
        return BatchSummary(
            "extract", str(path), str(output_root), downsample_factor,
            1, int(result.status == "ok"), int(result.status == "warning"),
            int(result.status == "error"), [result],
        )

    if path.is_dir():
        output_root = path.parent / f"{path.name}_extracted"
        log.info("output: %s", output_root)

        mat_files = _find_mat_files(path)
        if not mat_files:
            log.error("nessun file .mat trovato in: %s", path)
            return BatchSummary("extract", str(path), str(output_root), downsample_factor,
                                 0, 0, 0, 0, [])

        log.info("trovati  %d  file .mat in %s", len(mat_files), path)
        results = [
            _extract_single_file(
                mat_path, path, output_root,
                downsample_factor=downsample_factor, window_seconds=window_seconds,
            )
            for mat_path in mat_files
        ]

        n_ok = sum(r.status == "ok" for r in results)
        n_warning = sum(r.status == "warning" for r in results)
        n_error = sum(r.status == "error" for r in results)
        log.info("completato  totale=%d  ok=%d  warning=%d  errori=%d",
                  len(results), n_ok, n_warning, n_error)

        return BatchSummary("extract", str(path), str(output_root), downsample_factor,
                             len(results), n_ok, n_warning, n_error, results)

    raise ValueError(f"percorso non valido (ne' file .mat ne' cartella): {path}")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mat_tools",
        description=(
            "Strumenti di pre-elaborazione per il dataset Photovoltaic DC Arc "
            "Library: isolamento dei file .mat (clean) ed estrazione/"
            "sottocampionamento delle variabili fisiche (extract)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--summary-json", type=Path, default=None, metavar="PATH",
        help="salva un riepilogo strutturato dell'operazione in formato JSON",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_clean = sub.add_parser(
        "clean", help="copia solo i file .mat in <cartella>_clean/, replicando la struttura",
    )
    p_clean.add_argument("path", type=Path, help="cartella sorgente del dataset grezzo")
    p_clean.add_argument("--dry-run", action="store_true", help="simula senza copiare alcun file")

    p_extract = sub.add_parser(
        "extract", help="estrae le variabili fisiche e salva in <cartella>_extracted/",
    )
    p_extract.add_argument("path", type=Path, help="file .mat singolo o cartella sorgente")
    p_extract.add_argument(
        "--downsample-factor", type=int, default=DEFAULT_DOWNSAMPLE_FACTOR, metavar="N",
        help=(
            "tiene 1 campione ogni N "
            f"(default: {DEFAULT_DOWNSAMPLE_FACTOR}, {FS_ORIGINAL_HZ // 1000} kHz -> "
            f"{FS_TARGET_HZ // 1000} kHz)"
        ),
    )
    p_extract.add_argument(
        "--window", type=float, default=None, metavar="SECONDI",
        help="tronca ogni segnale a SECONDI dopo il sottocampionamento (opzionale)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "clean":
            summary = clean_directory(args.path, dry_run=args.dry_run)
        else:  # "extract"
            summary = extract_path(
                args.path,
                downsample_factor=args.downsample_factor,
                window_seconds=args.window,
            )
    except (NotADirectoryError, ValueError) as exc:
        log.error(str(exc))
        return 1

    if args.summary_json is not None:
        summary.to_json(args.summary_json)
        log.info("riepilogo salvato in: %s", args.summary_json)

    return 1 if summary.n_error > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
