"""Load the Tinker API key into this process's environment without printing it.

The key file holds one line (the key). Path: TINKER_KEY_FILE, default the user's Desktop file (reached through
/mnt/c under WSL, where tml-renderers has wheels; it has none for Windows).
"""
import os
from pathlib import Path

_DEFAULT = (r"C:\Users\patri\OneDrive\Desktop\tinker.txt.txt" if os.name == "nt"
            else "/mnt/c/Users/patri/OneDrive/Desktop/tinker.txt.txt")
KEY_FILE = Path(os.environ.get("TINKER_KEY_FILE", _DEFAULT))


def load_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if not key:
            raise SystemExit(f"empty key file {KEY_FILE}")
        os.environ["TINKER_API_KEY"] = key
    try:
        import truststore  # Norton re-signs HTTPS on this machine; trust the OS store
        truststore.inject_into_ssl()
    except ImportError:
        pass
