#!/usr/bin/env python3
"""Desktop launcher for the JoyAI-Video-Edit streaming demo.

    python deploy/jve_desktop.py --preset 16g      # 24 GB+ cards (5090/4090)
    python deploy/jve_desktop.py --preset 8g       # 8 GB cards, 840x480
    python deploy/jve_desktop.py --preset 8g-640   # 8 GB cards, 624x360 (faster)
    python deploy/jve_desktop.py --check           # print the plan, start nothing

What it does:

1. builds the environment for the chosen preset (text-stream stripping, weight
   offload, tight VRAM, block streaming for 8 GB cards),
2. starts `deploy/run_server.sh` as a child process and tails its log,
3. waits for `/health`,
4. shows the web UI in a native window (pywebview) or the default browser,
5. stops the server when the window closes / Ctrl-C.

Only the launcher is stdlib; pywebview is optional.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "deploy" / "run_server.sh"

# Presets: name -> (environment, resolution)
PRESETS = {
    "16g": {
        "env": {
            "JOYOMNI_LOW_VRAM": "1",
            "JOYOMNI_TEXT_KV": "1",
            "JOYOMNI_TEXT_OFFLOAD": "1",
            "JOYOMNI_TIGHT_VRAM": "1",
            "JOYOMNI_PE_BACKEND": "none",
        },
        "size": (840, 480),
        "note": "24 GB+: 15.3 GiB, ~0.31 s/chunk",
    },
    "8g": {
        "env": {
            "JOYOMNI_LOW_VRAM": "1",
            "JOYOMNI_TEXT_KV": "1",
            "JOYOMNI_TEXT_OFFLOAD": "1",
            "JOYOMNI_TIGHT_VRAM": "1",
            "JOYOMNI_STREAM_DIT": "1",
            "JOYOMNI_PE_BACKEND": "none",
        },
        "size": (840, 480),
        "note": "8 GB: 7.3 GiB reserved, blocks streamed from host RAM",
    },
    "8g-640": {
        "env": {
            "JOYOMNI_LOW_VRAM": "1",
            "JOYOMNI_TEXT_KV": "1",
            "JOYOMNI_TEXT_OFFLOAD": "1",
            "JOYOMNI_TIGHT_VRAM": "1",
            "JOYOMNI_STREAM_DIT": "1",
            "JOYOMNI_PE_BACKEND": "none",
        },
        "size": (624, 360),
        "note": "8 GB + lower resolution: fewer tokens, more fps",
    },
}
DEFAULT_PRESET = "16g"


def build_plan(args: argparse.Namespace) -> tuple[dict[str, str], int, int]:
    preset = PRESETS[args.preset]
    env = dict(preset["env"])
    width, height = preset["size"]
    if args.width:
        width = args.width
    if args.height:
        height = args.height
    # The VAE stem requires both dimensions to be multiples of 24.
    if width % 24 or height % 24:
        raise SystemExit(f"resolution {width}x{height} is invalid: both sides must be multiples of 24")
    env["JOYOMNI_WIDTH"] = str(width)
    env["JOYOMNI_HEIGHT"] = str(height)
    env["JOYOMNI_PORT"] = str(args.port)
    env["JOYOMNI_HOST"] = args.host
    if args.ckpt:
        env["JOYOMNI_DIT_CKPT"] = args.ckpt
    if args.precision:
        env["JOYOMNI_DIT_PRECISION"] = args.precision
    for item in args.env:
        key, _, value = item.partition("=")
        if not key:
            raise SystemExit(f"bad --env {item!r}, expected KEY=VALUE")
        env[key] = value
    return env, width, height


def print_plan(env: dict[str, str], width: int, height: int, args: argparse.Namespace) -> None:
    print(f"preset    : {args.preset}  ({PRESETS[args.preset]['note']})")
    print(f"resolution: {width}x{height}")
    print(f"url       : http://127.0.0.1:{args.port}/")
    print("environment:")
    for key in sorted(env):
        print(f"  {key}={env[key]}")
    print(f"launch    : bash {RUN_SCRIPT}")
    print(f"script    : {'found' if RUN_SCRIPT.is_file() else 'MISSING'}")
    print(f"python    : {sys.executable}")


def wait_health(port: int, timeout: float, process: subprocess.Popen) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    last_report = 0.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"[desktop] server exited with code {process.returncode}")
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8", "replace"))
                    print(f"[desktop] healthy: dit={Path(payload.get('dit_ckpt', '?')).name}")
                    return True
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            pass
        now = time.monotonic()
        if now - last_report > 15:
            print(f"[desktop] waiting for the server ({int(deadline - now)}s left)…", flush=True)
            last_report = now
        time.sleep(2)
    print("[desktop] timeout waiting for /health")
    return False


def show_window(url: str, title: str) -> None:
    try:
        import webview  # type: ignore
    except ImportError:
        print("[desktop] pywebview not installed -> opening the default browser")
        print("[desktop] (pip install pywebview for a real desktop window)")
        webbrowser.open(url)
        return
    webview.create_window(title, url, width=1280, height=800)
    webview.start()


def main() -> int:
    parser = argparse.ArgumentParser(description="JoyAI-Video-Edit desktop launcher")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--ckpt", default="", help="override JOYOMNI_DIT_CKPT")
    parser.add_argument("--precision", default="", help="override JOYOMNI_DIT_PRECISION")
    parser.add_argument("--timeout", type=float, default=900.0, help="health timeout (s)")
    parser.add_argument("--check", action="store_true", help="print the plan and exit")
    parser.add_argument("--no-window", action="store_true", help="run the server only")
    parser.add_argument("--log", default="", help="server log path (default: ./jve_desktop.log)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra environment override, repeatable")
    args = parser.parse_args()

    env, width, height = build_plan(args)
    print_plan(env, width, height, args)
    if args.check:
        return 0
    if not RUN_SCRIPT.is_file():
        print(f"[desktop] cannot find {RUN_SCRIPT}")
        return 2

    child_env = {**os.environ, **env, "PYTHONUNBUFFERED": "1"}
    # deploy/run_server.sh calls `python`; make sure the interpreter that runs the
    # launcher (usually a venv) is the one it finds.
    # NB: no resolve() - the venv's `python` is a symlink to /usr/bin/python3.10
    # and resolving it would put /usr/bin on PATH instead of the venv.
    python_dir = str(Path(sys.executable).parent)
    child_env["PATH"] = python_dir + os.pathsep + child_env.get("PATH", "")
    log_path = Path(args.log) if args.log else Path.cwd() / "jve_desktop.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    print(f"[desktop] server log -> {log_path}")
    process = subprocess.Popen(
        ["bash", str(RUN_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=child_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def stop(*_args) -> None:
        if process.poll() is None:
            print("[desktop] stopping the server…")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    signal.signal(signal.SIGINT, lambda *_: (stop(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(0)))

    try:
        if not wait_health(args.port, args.timeout, process):
            return 1
        url = f"http://127.0.0.1:{args.port}/"
        if args.no_window:
            print("[desktop] running headless; Ctrl-C stops the server")
            while process.poll() is None:
                time.sleep(1)
            return process.returncode or 0
        show_window(url, f"JoyAI-Video-Edit ({args.preset}, {width}x{height})")
        return 0
    finally:
        stop()
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
