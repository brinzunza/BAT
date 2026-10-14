"""Pull a consistent snapshot of PRODLIVETRADING's trades.db into analytics/prodlive/data/.

Source (first match wins):
  --src / PRODLIVE_DB       local path to a trades.db (e.g. ../PRODLIVETRADING/logs/practice/trades.db, or a copy you scp'd)
  --ssh / PRODLIVE_SSH      "user@host:/home/user/PRODLIVETRADING/logs/practice/trades.db"  (uses your ssh config/keys; PRODLIVE_SSH_OPTS for extra flags)
The remote DB is in WAL mode, so a plain `scp trades.db` can miss recent commits or tear; we run `sqlite3 .backup` (or python's sqlite3 backup API) on the
remote to produce a consistent file, copy that, and delete it. The bot is never touched (read-only). Local copies use the sqlite backup API for the same reason."""
import os, shlex, sqlite3, subprocess, shutil, time
from pathlib import Path
DATA = Path(__file__).parent / "data"; LATEST = DATA / "latest.db"
def _local_backup(src, dst):
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True); d = sqlite3.connect(dst)
    with d: s.backup(d)
    s.close(); d.close()
def _remote(spec):
    host, _, path = spec.partition(":")
    if not path: raise ValueError("PRODLIVE_SSH must look like user@host:/path/to/trades.db")
    opts = shlex.split(os.environ.get("PRODLIVE_SSH_OPTS", "")); tmp = f"/tmp/prodlive_snapshot_{os.getpid()}.db"
    py = f"import sqlite3;s=sqlite3.connect('file:{path}?mode=ro',uri=True);d=sqlite3.connect('{tmp}');s.backup(d);d.close()"
    subprocess.run(["ssh", *opts, host, "python3", "-c", shlex.quote(py)], check=True, timeout=120)
    DATA.mkdir(parents=True, exist_ok=True); out = DATA / "_incoming.db"
    try: subprocess.run(["scp", "-q", *opts, f"{host}:{tmp}", str(out)], check=True, timeout=300)
    finally: subprocess.run(["ssh", *opts, host, "rm", "-f", tmp], timeout=60)
    return out
def sync(src=None, ssh=None):
    DATA.mkdir(parents=True, exist_ok=True); src = src or os.environ.get("PRODLIVE_DB"); ssh = ssh or os.environ.get("PRODLIVE_SSH")
    if src:
        tmp = DATA / "_incoming.db"; _local_backup(str(Path(src).expanduser()), str(tmp))
    elif ssh: tmp = _remote(ssh)
    else: raise SystemExit("no source: set PRODLIVE_DB (local trades.db) or PRODLIVE_SSH (user@host:/path/trades.db), or pass --src/--ssh")
    c = sqlite3.connect(tmp); ok = c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"; tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}; c.close()
    if not ok or "trades" not in tabs: tmp.unlink(); raise SystemExit("snapshot failed integrity check or lacks a trades table; kept the previous latest.db")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()); keep = DATA / f"trades_{stamp}.db"; shutil.copy(tmp, keep); os.replace(tmp, LATEST)
    for old in sorted(DATA.glob("trades_*.db"))[:-20]: old.unlink()     # keep the last 20 snapshots
    return LATEST
