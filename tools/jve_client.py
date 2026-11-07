#!/usr/bin/env python3
"""Headless driver for the JoyAI-Video-Edit streaming server.

  python jve_client.py --ws ws://127.0.0.1:8080/ws --src face.jpg \
      --ref target.png --prompt "把画面中的人物替换成参考图中的人物" \
      --frames 96 --fps 24 --out runs/run1

Sends `start` JSON (prompt + base64 reference image), streams JPEG frames at the
requested rate, and stores every edited frame the server sends back.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
from pathlib import Path

import cv2
import websockets


def load_source(src: str, width: int, height: int):
    """Yield BGR frames: a video file decodes; an image repeats."""
    p = Path(src)
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"cannot read image {src}")
        img = cv2.resize(img, (width, height))
        while True:
            yield img
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {src}")
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("video has no frames")
        yield cv2.resize(frame, (width, height))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8080/ws")
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=840)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--detection", default="off", choices=["off", "near", "full"])
    ap.add_argument("--out", default="runs/live")
    ap.add_argument("--lockstep", action="store_true",
                    help="wait for each 8-frame chunk's output before sending the next")
    ap.add_argument("--profile", action="store_true", help="ask the server for per-chunk timings")
    ap.add_argument("--stall-timeout", type=float, default=90.0)
    ap.add_argument("--tail-wait", type=float, default=90.0)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    start = {
        "type": "start",
        "session_id": uuid.uuid4().hex,
        "prompt": args.prompt,
        "detection_mode": args.detection,
        "fps": args.fps,
        "input_codec": "mjpeg",
        "output_codec": "mjpeg",
        "output_quality": args.quality,
        "use_pe": False,
        # keep the wire protocol in camera mode even for file sources: file mode
        # switches the server to lossless upload and suppresses streamed output
        "source": "camera",
        "kv_reset_frames": 0,
    }
    if args.profile:
        start["profile_timings"] = True
    if args.ref:
        data = Path(args.ref).read_bytes()
        mime = "image/jpeg" if args.ref.lower().endswith((".jpg", ".jpeg")) else "image/png"
        start["ref_image"] = f"data:{mime};base64," + base64.b64encode(data).decode()

    got: list[float] = []
    started = time.monotonic()
    t_first: list[float] = []

    async with websockets.connect(args.ws, max_size=None, ping_interval=None) as ws:
        send_lock = asyncio.Lock()

        async def receiver():
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=args.tail_wait)
                except asyncio.TimeoutError:
                    return
                except websockets.ConnectionClosed:
                    return
                if isinstance(msg, bytes):
                    # binary payload == the JPEG announced by the previous output_frame
                    idx = len(got)
                    (outdir / f"out_{idx:04d}.jpg").write_bytes(msg)
                    got.append(time.monotonic())
                    if len(t_first) == 0:
                        t_first.append(time.monotonic() - started)
                    # the server's flow control needs this: without acks it treats
                    # the client as congested and drops whole chunks
                    async with send_lock:
                        try:
                            await ws.send(json.dumps({"type": "ack", "recv": len(got),
                                                      "session_id": start["session_id"]}))
                        except Exception:
                            return
                    continue
                try:
                    payload = json.loads(msg)
                except Exception:
                    continue
                t = payload.get("type")
                if t in ("started", "ready", "chunk_start"):
                    if t != "chunk_start":
                        print("[srv]", json.dumps(payload)[:300], flush=True)
                elif t == "error":
                    print("[ERR]", payload.get("message"), flush=True)

        recv_task = asyncio.create_task(receiver())
        async with send_lock:
            await ws.send(json.dumps(start))
        print(f"[cli] start sent (ref={'yes' if args.ref else 'no'}, {args.frames} frames @ {args.fps}fps)",
              flush=True)

        gen = load_source(args.src, args.width, args.height)
        t0 = time.monotonic()
        for i in range(args.frames):
            frame = next(gen)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            if not ok:
                continue
            if args.lockstep and i > 0 and (i - 1) % 8 == 0:
                # start of a new chunk: only send once every earlier frame was edited
                # back, otherwise the server's ~8-frame input buffer overflows
                while len(got) < i:
                    if time.monotonic() - t0 > args.stall_timeout:
                        print(f"[cli] stall at frame {i} (got {len(got)})", flush=True)
                        break
                    await asyncio.sleep(0.01)
            else:
                target = t0 + i / args.fps
                now = time.monotonic()
                if target > now:
                    await asyncio.sleep(target - now)
                await asyncio.sleep(0)
            async with send_lock:
                await ws.send(buf.tobytes())
        send_done = time.monotonic()
        print(f"[cli] sent {args.frames} frames in {send_done - t0:.2f}s "
              f"({args.frames / (send_done - t0):.1f} fps wall)", flush=True)

        # keep reading until the server goes quiet
        try:
            await asyncio.wait_for(recv_task, timeout=args.tail_wait)
        except asyncio.TimeoutError:
            recv_task.cancel()

    total = time.monotonic() - started
    span = (got[-1] - got[0]) if len(got) > 1 else 0.0
    first = f"{t_first[0]:.1f}s" if t_first else "n/a"
    rate = ((len(got) - 1) / span) if span > 0 else 0.0
    print(f"[cli] received {len(got)} edited frames in {total:.1f}s"
          f" | first frame after {first} | steady {rate:.2f} fps", flush=True)
    print(json.dumps({"frames_out": len(got), "first_out_s": t_first[0] if t_first else None,
                      "fps": ((len(got) - 1) / span) if span > 0 else None, "wall_s": total}))


if __name__ == "__main__":
    asyncio.run(main())
