"""Experiment metadata persistence for reproducible research.

Every research run should produce an ExperimentRecord that captures all
configuration, runtime context, and outcomes in a single JSON file.
This enables exact reproduction of any past experiment.

Public API
----------
ExperimentRecord
build_experiment_record
save_experiment
load_experiment
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

from predictor.research.errors import ResearchError


class ExperimentPersistenceError(ResearchError):
    """Raised when experiment record cannot be saved or loaded."""


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    """Complete metadata record for one research experiment run.

    Attributes
    ----------
    experiment_id : str
        UUID4 unique identifier.
    timestamp : str
        ISO 8601 UTC timestamp of experiment start.
    git_hash : str
        Short git commit hash, or "unknown" if not in a git repo.
    runtime_seconds : float
        Total wall-clock execution time.
    random_seed : int
        Base random seed used for all permutation tests.
    validation_config : Dict
        Serialised ResearchValidationConfig fields.
    symbol_universe : Tuple[str, ...]
        All symbols included in the run.
    strategy_names : Tuple[str, ...]
        All strategy names in the chosen universe.
    permutation_iterations : int
        Number of permutation iterations per test.
    fold_structure : Dict
        Walk-forward fold parameters (train, test, step windows).
    generated_plots : Tuple[str, ...]
        Absolute paths of all PNG files generated during this run.
    rejection_summary : Dict
        Counts of each rejection reason and overall pass rate.
    notes : str
        Free-form notes (CLI flags, experiment intent, etc.).
    """

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_hash: str = "unknown"
    runtime_seconds: float = 0.0
    random_seed: int = 0
    validation_config: Dict = field(default_factory=dict)
    symbol_universe: Tuple[str, ...] = field(default_factory=tuple)
    strategy_names: Tuple[str, ...] = field(default_factory=tuple)
    permutation_iterations: int = 0
    fold_structure: Dict = field(default_factory=dict)
    generated_plots: Tuple[str, ...] = field(default_factory=tuple)
    rejection_summary: Dict = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_git_hash() -> str:
    """Return short git hash of HEAD, or 'unknown'."""
    try:
        raw = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return raw.decode("ascii").strip()
    except Exception:
        return "unknown"


def build_experiment_record(
    *,
    runtime_seconds: float,
    random_seed: int,
    validation_config: object,
    symbol_universe: Tuple[str, ...],
    strategy_names: Tuple[str, ...],
    generated_plots: Tuple[str, ...] = (),
    rejection_counts: Dict[str, int] | None = None,
    total_runs: int = 0,
    passed_runs: int = 0,
    notes: str = "",
) -> ExperimentRecord:
    """Build an ExperimentRecord from a completed research run.

    Parameters
    ----------
    runtime_seconds : float
        Total wall-clock time from run start to finish.
    random_seed : int
        Base random seed used.
    validation_config : ResearchValidationConfig
        Config object — will be serialised to dict.
    symbol_universe : Tuple[str, ...]
        All symbols in the run.
    strategy_names : Tuple[str, ...]
        All strategy names in the run.
    generated_plots : Tuple[str, ...]
        Paths to generated PNG files.
    rejection_counts : Dict[str, int] | None
        {reason: count} of all rejection reasons observed.
    total_runs : int
        Total (symbol × strategy) pairs evaluated.
    passed_runs : int
        Number of pairs that passed all validation stages.
    notes : str
        Free-form experiment notes.

    Returns
    -------
    ExperimentRecord
    """
    config_dict = {}
    if validation_config is not None:
        try:
            config_dict = {
                k: v for k, v in vars(validation_config).items()
                if not k.startswith("_")
            }
        except TypeError:
            # dataclass — use __dataclass_fields__
            import dataclasses
            config_dict = {
                f.name: getattr(validation_config, f.name)
                for f in dataclasses.fields(validation_config)
            }

    rejection_summary: Dict = dict(rejection_counts or {})
    if total_runs > 0:
        rejection_summary["pass_rate"] = passed_runs / total_runs
        rejection_summary["total_runs"] = total_runs
        rejection_summary["passed_runs"] = passed_runs

    fold_structure = {
        "train_window": config_dict.get("train_window"),
        "test_window": config_dict.get("test_window"),
        "walk_forward_step": config_dict.get("walk_forward_step"),
    }

    return ExperimentRecord(
        git_hash=_resolve_git_hash(),
        runtime_seconds=round(runtime_seconds, 3),
        random_seed=random_seed,
        validation_config=config_dict,
        symbol_universe=tuple(symbol_universe),
        strategy_names=tuple(strategy_names),
        permutation_iterations=int(config_dict.get("permutation_iterations", 0)),
        fold_structure=fold_structure,
        generated_plots=tuple(str(p) for p in generated_plots),
        rejection_summary=rejection_summary,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _serialise(obj: object) -> object:
    """Recursively convert non-JSON-serialisable types."""
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_experiment(
    record: ExperimentRecord,
    output_dir: Path | str,
) -> Path:
    """Write ExperimentRecord to a JSON file in output_dir.

    Filename: experiment_{record.experiment_id[:8]}.json

    Returns
    -------
    Path
        Path of the written file.

    Raises
    ------
    ExperimentPersistenceError
        If the file cannot be written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filename = out / f"experiment_{record.experiment_id[:8]}.json"

    try:
        payload = _serialise(asdict(record))
        filename.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        raise ExperimentPersistenceError(
            f"Failed to write experiment record to {filename}: {exc}"
        ) from exc

    return filename


def load_experiment(path: Path | str) -> ExperimentRecord:
    """Load an ExperimentRecord from a JSON file.

    Returns
    -------
    ExperimentRecord

    Raises
    ------
    ExperimentPersistenceError
        If the file cannot be read or parsed.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except Exception as exc:
        raise ExperimentPersistenceError(
            f"Failed to load experiment record from {p}: {exc}"
        ) from exc

    # Convert list fields back to tuples
    for key in ("symbol_universe", "strategy_names", "generated_plots"):
        if key in raw and isinstance(raw[key], list):
            raw[key] = tuple(raw[key])

    return ExperimentRecord(**{k: v for k, v in raw.items() if k in ExperimentRecord.__dataclass_fields__})
