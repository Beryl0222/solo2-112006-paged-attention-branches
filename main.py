"""Compatibility alias — canonical implementation in :mod:`paged_attention`."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paged_attention as _pa  # noqa: E402
from paged_attention import *  # noqa: E402,F401,F403

globals().update({k: v for k, v in vars(_pa).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_pa.main())
