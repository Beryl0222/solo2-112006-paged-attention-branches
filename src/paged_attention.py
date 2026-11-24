"""Compatibility shim so ``src.paged_attention`` resolves to the root module."""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import paged_attention as _pa  # noqa: E402
from paged_attention import *  # noqa: E402,F401,F403

globals().update({k: v for k, v in vars(_pa).items() if not k.startswith("__")})
