#!/usr/bin/env bash
# Run a script with the WSL Tinker venv: wsl.exe -d Ubuntu-24.04 -u root --exec bash wpy.sh <script.py> [args]
# INKY_VENV overrides the venv (default /root/inky-tinker/venv).
exec "${INKY_VENV:-/root/inky-tinker/venv}/bin/python" "$@"
