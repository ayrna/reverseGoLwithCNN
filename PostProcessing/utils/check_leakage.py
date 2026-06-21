"""
check_leakage.py  (v2 — formato CSR sparse)
════════════════════════════════════════════
Detecta data leakage entre ConwayStates/Test y Train/Validation.

Formato de los .npz
───────────────────
Cada fichero es una matriz dispersa CSR de shape (N, 3*M):
    columnas  0 :  M  → tableros base
    columnas  M : 2M  → tableros iniciales  (start_i)
    columnas 2M : 3M  → tableros finales    (stop_i)

Cada fila es un sample (tablero aplanado H*W).
El leakage se comprueba sobre iniciales Y finales por separado.

Estructura esperada:
    parent_folder/
        └── {shape}_{delta}_{seed}/
            └── ConwayStates/
                ├── Train/       └── {model_name}/  *.npz
                ├── Validation/  └── {model_name}/  *.npz
                └── Test/                           *.npz

Uso:
    python check_leakage.py \\
        --parent_folder /ruta \\
        --shape 15x15 --delta 1 --seed 221 --model_name Classic

    # O sólo iniciales/finales:
        --check start   (sólo tableros iniciales)
        --check stop    (sólo tableros finales)
        --check both    (por defecto)
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse


# ─── Carga de un .npz en el formato CSR del proyecto ─────────────────────────

def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga un .npz con formato CSR y devuelve:
        iniciales  (N, H*W)  — tableros start_i  binarios uint8
        finales    (N, H*W)  — tableros stop_i   binarios uint8
    """
    npz_file = np.load(path, allow_pickle=True)

    X = sparse.csr_matrix(
        (npz_file["data"], npz_file["indices"], npz_file["indptr"]),
        shape=npz_file["shape"],
    )
    M = int(npz_file["M"].item())

    iniciales = (X[:, M  : 2*M].toarray() > 0).astype(np.uint8)
    finales   = (X[:, 2*M: 3*M].toarray() > 0).astype(np.uint8)

    return iniciales, finales


# ─── Hashing de tableros ──────────────────────────────────────────────────────

def hash_row(row: np.ndarray) -> str:
    """SHA-256 de una fila (H*W,). Rápido y sin colisiones prácticas."""
    return hashlib.sha256(row.tobytes()).hexdigest()


# ─── Indexación de un split ───────────────────────────────────────────────────

def index_split(
    folder: Path,
    check: str,
) -> dict[str, list[tuple[Path, int, str]]]:
    """
    Recorre todos los .npz de `folder` y construye:
        { sha256 → [(path_fichero, fila_idx, "start"|"stop"), ...] }
    """
    if not folder.exists():
        print(f"  [WARN] No existe: {folder}", file=sys.stderr)
        return {}

    npz_files = sorted(folder.rglob("*.npz"))
    if not npz_files:
        print(f"  [WARN] Sin .npz en: {folder}", file=sys.stderr)
        return {}

    index: dict[str, list] = defaultdict(list)
    total_rows = 0

    for path in npz_files:
        try:
            iniciales, finales = load_npz(path)
            n = iniciales.shape[0]
            total_rows += n

            if check in ("start", "both"):
                for i in range(n):
                    h = hash_row(iniciales[i])
                    index[h].append((path, i, "start"))

            if check in ("stop", "both"):
                for i in range(n):
                    h = hash_row(finales[i])
                    index[h].append((path, i, "stop"))

        except Exception as e:
            print(f"  [ERROR] {path.name}: {e}", file=sys.stderr)

    print(f"      {len(index):,} hashes únicos  |  "
          f"{total_rows:,} tableros  |  "
          f"{len(npz_files)} fichero(s)")

    return dict(index)


# ─── Análisis de leakage ──────────────────────────────────────────────────────

def check_leakage(
    parent_folder : str,
    shape         : str,
    delta         : int | str,
    seed          : int | str,
    model_name    : str,
    check         : str  = "both",
    verbose       : bool = True,
) -> dict:
    """
    Parámetros
    ----------
    check : "start" | "stop" | "both"
        Qué tipo de tableros comparar.

    Returns
    -------
    dict con métricas y lista de leaks detallados.
    """
    base = Path(parent_folder) / f"{shape}_{delta}_{seed}" / "ConwayStates"

    train_folder = base / "Train"      / model_name
    val_folder   = base / "Validation" / model_name
    test_folder  = base / "Test"

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  Experimento : {shape}_{delta}_{seed}")
        print(f"  Modelo      : {model_name}")
        print(f"  Comparando  : tableros '{check}'")
        print(f"  Base path   : {base}")
        print(f"{'═'*65}")

    # ── Indexación ────────────────────────────────────────────────────────────
    print("\n[1/3] Indexando Train...")
    train_index = index_split(train_folder, check)

    print("[2/3] Indexando Validation...")
    val_index   = index_split(val_folder,   check)

    print("[3/3] Indexando Test...")
    test_index  = index_split(test_folder,  check)

    if not test_index:
        print("\n  [ERROR] Test vacío, no hay nada que comparar.")
        return {}

    # ── Comparación ───────────────────────────────────────────────────────────
    leaked: list[dict] = []

    leaked_in_train = 0
    leaked_in_val   = 0

    # total de entradas en test (puede ser 2×N si check=="both")
    total_test_entries = sum(len(v) for v in test_index.values())

    for h, test_entries in test_index.items():
        in_train = h in train_index
        in_val   = h in val_index

        if not in_train and not in_val:
            continue

        for (t_path, t_row, t_kind) in test_entries:
            leaked.append({
                "test_file"      : str(t_path),
                "row"            : t_row,
                "kind"           : t_kind,
                "hash"           : h,
                "leaked_in_train": in_train,
                "leaked_in_val"  : in_val,
                "train_sources"  : [
                    {"file": str(p), "row": r, "kind": k}
                    for (p, r, k) in train_index.get(h, [])[:3]
                ],
                "val_sources"    : [
                    {"file": str(p), "row": r, "kind": k}
                    for (p, r, k) in val_index.get(h, [])[:3]
                ],
            })

        if in_train: leaked_in_train += len(test_entries)
        if in_val:   leaked_in_val   += len(test_entries)

    # Normalizar al número de tableros (no de entradas hash)
    # Para "both": cada tablero genera 2 entradas → dividir entre 2
    divisor    = 2 if check == "both" else 1
    total_test = total_test_entries // divisor
    n_leaked   = len(set(
        (e["test_file"], e["row"]) for e in leaked
    ))

    def pct(n): return (n / total_test * 100) if total_test > 0 else 0.0

    results = {
        "total_test"        : total_test,
        "n_leaked_boards"   : n_leaked,
        "leaked_in_train"   : leaked_in_train // divisor,
        "leaked_in_val"     : leaked_in_val   // divisor,
        "pct_leaked_train"  : pct(leaked_in_train // divisor),
        "pct_leaked_val"    : pct(leaked_in_val   // divisor),
        "pct_leaked_any"    : pct(n_leaked),
        "leaked_detail"     : leaked,
    }

    if verbose:
        _print_report(results, check)

    return results


# ─── Reporte ──────────────────────────────────────────────────────────────────

def _print_report(r: dict, check: str) -> None:
    total = r["total_test"]

    print(f"\n{'─'*65}")
    print(f"  RESUMEN DE DATA LEAKAGE  (comparando: {check})")
    print(f"{'─'*65}")
    print(f"  Tableros en Test                  : {total:>10,}")
    print(f"  Tableros leakeados en Train        : {r['leaked_in_train']:>10,}  "
          f"({r['pct_leaked_train']:5.2f} %)")
    print(f"  Tableros leakeados en Validation   : {r['leaked_in_val']:>10,}  "
          f"({r['pct_leaked_val']:5.2f} %)")
    print(f"  Tableros leakeados (train o val)   : {r['n_leaked_boards']:>10,}  "
          f"({r['pct_leaked_any']:5.2f} %)")
    print(f"{'─'*65}")

    if r["n_leaked_boards"] == 0:
        print("  ✅  Sin leakage detectado.\n")
        return

    pct = r["pct_leaked_any"]
    severity = ("🟡  BAJO" if pct < 5 else
                "🟠  MODERADO" if pct < 20 else
                "🔴  ALTO")
    print(f"  Severidad : {severity}  ({pct:.2f} % del test contaminado)")
    print(f"{'─'*65}")

    # Detalle de hasta 10 tableros leakeados
    seen = set()
    shown = 0
    print(f"\n  Ejemplos de leaks (máx. 10):\n")
    for e in r["leaked_detail"]:
        key = (e["test_file"], e["row"])
        if key in seen:
            continue
        seen.add(key)
        flags = []
        if e["leaked_in_train"]: flags.append("TRAIN")
        if e["leaked_in_val"]:   flags.append("VAL")
        print(f"  [{'+'.join(flags)}]  {Path(e['test_file']).name}  "
              f"fila {e['row']}  ({e['kind']})")
        for src in e["train_sources"][:1]:
            print(f"         ← train: {Path(src['file']).name}  fila {src['row']}")
        for src in e["val_sources"][:1]:
            print(f"         ← val  : {Path(src['file']).name}  fila {src['row']}")
        shown += 1
        if shown >= 10:
            break

    remaining = r["n_leaked_boards"] - shown
    if remaining > 0:
        print(f"\n  ... y {remaining:,} tableros más en results['leaked_detail']")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent_folder", required=True)
    p.add_argument("--shape",         required=True)
    p.add_argument("--delta",         required=True)
    p.add_argument("--seed",          required=True)
    p.add_argument("--model_name",    required=True)
    p.add_argument("--check",         default="both",
                   choices=["start", "stop", "both"],
                   help="Qué tableros comparar (default: both)")
    args = p.parse_args()

    check_leakage(
        parent_folder = args.parent_folder,
        shape         = args.shape,
        delta         = int(args.delta),
        seed          = int(args.seed),
        model_name    = args.model_name,
        check         = args.check,
    )
