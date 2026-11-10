"""ML pipeline helpers: runs, checkpoints, configs and subprocess job runners.

Training and evaluation run as ``python train.py`` / ``python evaluate.py``
subprocesses so the UI stays responsive, jobs can be cancelled, and the
existing scripts remain the single source of truth for how a run is made.
"""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILES = {"data": ROOT / "configs" / "data.yaml", "model": ROOT / "configs" / "model.yaml",
                "train": ROOT / "configs" / "training.yaml"}
MODELS = ("mlp", "cnn", "lstm", "transformer")
LOSSES = ("bce", "mse", "huber")
OPTIMIZERS = ("adam", "adamw", "sgd")
SCHEDULERS = ("none", "cosine", "step", "plateau")


def load_configs() -> Dict[str, Dict[str, Any]]:
    return {k: yaml.safe_load(open(p)) or {} for k, p in CONFIG_FILES.items()}


@dataclass
class RunInfo:
    name: str
    path: Path
    epochs: int
    best_val: float
    best_epoch: int
    last_train: float
    last_val: float
    mtime: float
    kind: str          # "train" | "folds"


def read_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    f = Path(run_dir) / "metrics.jsonl"
    rows = []
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def list_runs(run_dir: Optional[Path] = None) -> List[RunInfo]:
    base = Path(run_dir or ROOT / "runs")
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        f = d / "metrics.jsonl"
        if not (d.is_dir() and f.exists()):
            continue
        rows = read_metrics(d)
        tr = [r for r in rows if "val_loss" in r]
        if tr:
            best = min(tr, key=lambda r: r["val_loss"])
            out.append(RunInfo(d.name, d, len(tr), best["val_loss"], int(best.get("epoch", 0)),
                               tr[-1].get("train_loss", float("nan")), tr[-1]["val_loss"], f.stat().st_mtime, "train"))
        elif rows:
            out.append(RunInfo(d.name, d, len(rows), float("nan"), 0, float("nan"), float("nan"), f.stat().st_mtime, "folds"))
    return sorted(out, key=lambda r: r.mtime, reverse=True)


@dataclass
class CheckpointInfo:
    path: Path
    name: str
    size: int
    mtime: float


def list_checkpoints(ckpt_dir: Optional[Path] = None) -> List[CheckpointInfo]:
    base = Path(ckpt_dir or ROOT / "checkpoints")
    if not base.is_dir():
        return []
    return sorted((CheckpointInfo(p, p.name, p.stat().st_size, p.stat().st_mtime) for p in base.glob("*.pt")),
                  key=lambda c: c.mtime, reverse=True)


def checkpoint_config(path: Path) -> Dict[str, Any]:
    """Read the config stored in a checkpoint. Only open checkpoints you trust: they are pickles."""
    from src.utils.checkpoint import load_checkpoint
    ck = load_checkpoint(str(path))
    return {"cfg": ck.get("cfg", {}), "extra": ck.get("extra", {})}


def overrides_from(cfgs: Dict[str, Dict[str, Any]], base: Dict[str, Dict[str, Any]]) -> List[str]:
    """Only pass values that differ from the config files (keeps the command readable)."""
    out = []
    for sec, vals in cfgs.items():
        for k, v in vals.items():
            if base.get(sec, {}).get(k) != v:
                out.append(f"{sec}.{k}={yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}")
    return out


class Job:
    """A cancellable subprocess whose output is streamed line by line."""

    def __init__(self, argv: List[str], on_line: Callable[[str], None], on_done: Callable[[int], None],
                 cwd: Path = ROOT):
        self.argv, self.on_line, self.on_done, self.cwd = argv, on_line, on_done, cwd
        self.proc: Optional[subprocess.Popen] = None
        self.returncode: Optional[int] = None
        self.started = 0.0
        self._cancelled = False

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(self.cwd)}
        self.started = time.time()
        self.proc = subprocess.Popen(self.argv, cwd=str(self.cwd), env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            if line:
                self.on_line(line)
        self.returncode = self.proc.wait()
        self.on_done(-15 if self._cancelled else self.returncode)

    def cancel(self) -> None:
        if self.running:
            self._cancelled = True
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()


def train_argv(name: str, overrides: List[str]) -> List[str]:
    argv = [sys.executable, "train.py", "--name", name]
    if overrides:
        argv += ["--override", *overrides]
    return argv


def evaluate_argv(ckpt: str, walk_forward: bool, cost: float, folds: int) -> List[str]:
    argv = [sys.executable, "evaluate.py", "--ckpt", ckpt]
    if walk_forward:
        argv += ["--walk-forward", "--cost", str(cost), "--folds", str(folds)]
    return argv


def parse_result_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """evaluate.py prints one dict literal per fold / result."""
    rows = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                rows.append(ast.literal_eval(ln))
            except (ValueError, SyntaxError):
                pass
    return rows
