"""API keys and settings from a plain text file, for HPC or local runs (no Colab secrets there).

The file holds one setting per line, in either form (comments with #, blank lines ignored):

    GEMINI API KEY: AIza...            # "label: value", as you'd write it by hand
    OPENROUTER_API_KEY=sk-or-...       # or NAME=value (.env style; "export NAME=value" works too)

Where it is looked for, first match wins:
  1. the path given (--keys-file, or OperaAgent(keys_file=...))
  2. $OPERA_KEYS_FILE
  3. ./api_keys.txt                      (the folder you run from; git-ignored in this repo)
  4. ~/.config/opera/api_keys.txt        (your home directory: good on a shared HPC file system)

Values already set in the environment win over the file, so a job script can still override a key.
Keep the file private: on a shared system `chmod 600 api_keys.txt` (a warning is printed otherwise).
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

# setting name → the names people tend to write for it
KNOWN = {
    "GEMINI_API_KEY": {"GEMINI_API_KEY", "GEMINI_KEY", "GEMINI", "GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY"},
    "OPENROUTER_API_KEY": {"OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPENROUTER", "OPEN_ROUTER_API_KEY"},
    "GEMINI_MODEL": {"GEMINI_MODEL"},
    "OPENROUTER_MODEL": {"OPENROUTER_MODEL", "QWEN_MODEL"},
    "OPENROUTER_BASE_URL": {"OPENROUTER_BASE_URL"},
    "OPERA_YOLO_MODEL": {"OPERA_YOLO_MODEL", "YOLO_MODEL"},
}
_ALIAS = {alias: name for name, aliases in KNOWN.items() for alias in aliases}
DEFAULT_LOCATIONS = (Path("api_keys.txt"), Path.home() / ".config" / "opera" / "api_keys.txt")


def _normalise(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.strip().upper()).strip("_")


def parse_keys_file(path: str | Path) -> dict[str, str]:
    """{setting name: value} from a key file. Unknown labels raise, so a typo doesn't go unnoticed."""
    out = {}
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        m = re.match(r"^([^:=]+?)\s*[:=]\s*(.*)$", line)
        if not m:
            raise ValueError(f"{path}:{n}: expected 'NAME: value' or 'NAME=value'")
        label, value = m.group(1), m.group(2).strip()
        if value and value[0] not in "'\"" and " #" in value:     # trailing comment
            value = value.split(" #", 1)[0].strip()
        value = value.strip("'\"")
        name = _ALIAS.get(_normalise(label))
        if name is None:
            raise ValueError(f"{path}:{n}: unknown setting {label.strip()!r} "
                             f"(use one of: {', '.join(sorted(KNOWN))})")
        if value:
            out[name] = value
    return out


def find_keys_file(path: str | Path | None = None) -> Path | None:
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"keys file not found: {p}")
        return p
    env = os.environ.get("OPERA_KEYS_FILE")
    if env:
        return find_keys_file(env)
    return next((p for p in DEFAULT_LOCATIONS if p.is_file()), None)


def load_keys(path: str | Path | None = None, override: bool = False, log=print) -> list[str]:
    """Put the settings from the key file into os.environ (without overwriting ones already set, unless
    override=True). Returns the names that were set. No file found → nothing happens."""
    p = find_keys_file(path)
    if p is None:
        return []
    if os.name == "posix" and p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        log(f"warning: {p} can be read by other users; run: chmod 600 {p}")
    loaded = []
    for name, value in parse_keys_file(p).items():
        if override or not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    if loaded:
        log(f"settings from {p}: {', '.join(loaded)}")            # names only, never the values
    return loaded


def pop_keys_file_arg(argv: list[str]) -> str | None:
    """Remove '--keys-file PATH' / '--keys-file=PATH' from argv (it applies to every subcommand)."""
    for i, a in enumerate(argv):
        if a == "--keys-file":
            if i + 1 >= len(argv):
                raise SystemExit("--keys-file needs a path")
            path = argv[i + 1]
            del argv[i:i + 2]
            return path
        if a.startswith("--keys-file="):
            del argv[i]
            return a.split("=", 1)[1]
    return None
