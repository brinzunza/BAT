"""Guard: no credential-looking literals in tracked files (keys belong in .env, which is git-ignored)."""
import re, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [r"PK[A-Z0-9]{18}\b", r"\b[0-9a-f]{32}-[0-9a-f]{32}\b", r"AKIA[0-9A-Z]{16}", r"sk-[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{30,}", r"xox[bap]-[A-Za-z0-9-]{10,}", r"AIza[0-9A-Za-z_\-]{35}", r"-----BEGIN [A-Z ]*PRIVATE KEY",
            r"""(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*['"][A-Za-z0-9_\-]{20,}['"]""", r"(?<![\w-])\d{3}-\d{3}-\d{7,8}-\d{3}(?![\w-])"]   # last = OANDA account id
SKIP = (".json", ".jsonl", ".csv", ".png", ".pdf", ".so", ".parquet", ".pkl", ".joblib")
def test_no_secret_literals_in_tracked_files():
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout.split("\0"); bad = []
    for f in files:
        if not f or f.endswith(SKIP) or f.startswith("tradingResearch/"): continue
        try: t = (ROOT / f).read_text(errors="ignore")
        except OSError: continue
        for p in PATTERNS:
            if re.search(p, t): bad.append((f, p))
    assert not bad, f"possible credentials committed: {bad}"
