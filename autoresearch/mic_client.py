"""Headless client for the microphone route: stream a wav at real-time pace to /ws/mic (or straight to the ASR realtime
endpoint with --asr) and print the events. Used to test the live path without a person at the microphone.

  python -m autoresearch.mic_client data/primock57/mixed/day1_consultation01.wav --seconds 90 --model qwen27b --interval 15
  python -m autoresearch.mic_client file.wav --asr ws://beast:8090/v1/realtime --seconds 60      # ASR only, prints delta latency
"""
import argparse, asyncio, base64, json, sys, time, wave

import numpy as np
import websockets


def load_pcm16(path: str) -> bytes:
    w = wave.open(path, "rb")
    sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    raw = w.readframes(n); w.close()
    a = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.uint8).astype(np.float32)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        idx = np.arange(0, len(a), sr / 16000.0)
        a = np.interp(idx, np.arange(len(a)), a)
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


async def run(a):
    pcm = load_pcm16(a.wav)
    if a.seconds:
        pcm = pcm[: int(a.seconds * 32000)]
    chunk = int(0.1 * 32000)  # 100 ms
    t0 = time.time(); words = 0; first = None
    if a.asr:
        async with websockets.connect(a.asr, max_size=None) as ws:
            await ws.send(json.dumps({"type": "session.update", "model": a.asr_model}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            async def reader():
                nonlocal words, first
                async for raw in ws:
                    ev = json.loads(raw)
                    if ev.get("type") == "transcription.delta":
                        if first is None:
                            first = time.time() - t0
                        words += len(ev["delta"].split())
                        print(f"[{time.time()-t0:6.1f}s] {ev['delta']}", flush=True)
                    elif ev.get("type") == "transcription.done":
                        print(f"DONE {len(ev.get('text','').split())} words", flush=True); return
                    elif ev.get("type") == "error":
                        print("ERR", ev, flush=True)
            rt = asyncio.create_task(reader())
            for i in range(0, len(pcm), chunk):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + chunk]).decode()}))
                await asyncio.sleep(max(0, (i + chunk) / 32000 - (time.time() - t0)))
            sent = time.time() - t0
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            try:
                await asyncio.wait_for(rt, timeout=30)
            except asyncio.TimeoutError:
                print("timeout waiting for done")
            print(f"audio {len(pcm)/32000:.1f}s sent in {sent:.1f}s; first delta at {first}; {words} words", flush=True)
        return
    url = f"{a.url}/ws/mic?model={a.model}&interval={a.interval}"
    async with websockets.connect(url, max_size=None) as ws:
        async def reader():
            async for raw in ws:
                ev = json.loads(raw); e = ev.get("event")
                if e == "transcript":
                    print(f"[{ev['t']:6.1f}] {ev['text']}" + (f"  FLAGS {ev['flags']}" if ev.get("flags") else ""), flush=True)
                elif e == "synthesis":
                    s = ev["synth"]; print(f"=== view {ev['snapshot']+1} at {ev['t']}s, {ev['latency_s']}s: complaint={s.get('complaint')!r} dx={[d.get('dx') for d in s.get('differential') or [] if isinstance(d, dict)]} next={s.get('next_question')!r} flags={len(s.get('red_flags') or [])}", flush=True)
                elif e in ("activity",):
                    print(f"   · {ev['t']:6.1f} {ev['task']} {ev['status']}: {ev['detail'][:100]}", flush=True)
                else:
                    print("##", json.dumps(ev)[:300], flush=True)
                if e == "final":
                    return
        rt = asyncio.create_task(reader())
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk])
            await asyncio.sleep(max(0, (i + chunk) / 32000 - (time.time() - t0)))
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(rt, timeout=120)
        except asyncio.TimeoutError:
            print("timeout waiting for final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav"); ap.add_argument("--seconds", type=float, default=0); ap.add_argument("--url", default="ws://localhost:8700")
    ap.add_argument("--model", default="qwen27b"); ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--asr", default=""); ap.add_argument("--asr_model", default="voxtral-realtime")
    asyncio.run(run(ap.parse_args()))
