import argparse
import base64
import hashlib
import os
import subprocess
import sys
import time

import requests
from openai import OpenAI

from shared_runner import CONDITIONS, run_evaluation

_HERE = os.path.dirname(os.path.abspath(__file__))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set. Get one at https://openrouter.ai/keys")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SEMANTIC_MATCHER_URL = os.getenv("SEMANTIC_MATCHER_URL", "http://127.0.0.1:8002")

DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(_HERE, os.pardir, "dataset", "Games_json"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(_HERE, os.pardir, "runs"))
COMPRESS_DIR = os.getenv("COMPRESS_DIR", os.path.join(OUTPUT_DIR, "_inline_videos"))

DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
SIZE_LIMIT_BYTES = int(os.getenv("INLINE_LIMIT_MB", "14")) * 1024 * 1024
MAX_TOKENS = int(os.getenv("OR_MAX_TOKENS", "4096"))


client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def _ffmpeg_compress(src: str, dst: str, scale: int, crf: int) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"scale='min({scale},iw)':-2",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "64k",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception as e:
        print(f"[WARN] ffmpeg failed ({scale}p crf{crf}): {e}")
        return False


def compress_if_needed(video_path: str) -> str:
    """Return a path under the inline size limit, transcoding (cached) if needed."""
    if os.path.getsize(video_path) <= SIZE_LIMIT_BYTES:
        return video_path

    os.makedirs(COMPRESS_DIR, exist_ok=True)
    h = hashlib.md5(video_path.encode()).hexdigest()[:12]
    dst = os.path.join(COMPRESS_DIR, f"{h}.mp4")
    if os.path.exists(dst) and 0 < os.path.getsize(dst) <= SIZE_LIMIT_BYTES:
        return dst

    for scale, crf in [(1280, 30), (960, 32), (720, 34), (480, 36)]:
        if _ffmpeg_compress(video_path, dst, scale, crf):
            sz = os.path.getsize(dst)
            if sz <= SIZE_LIMIT_BYTES:
                print(f"    [COMPRESS] {os.path.basename(video_path)} "
                      f"{os.path.getsize(video_path)/1048576:.1f}MB -> {sz/1048576:.1f}MB "
                      f"({scale}p crf{crf})")
                return dst
    print(f"[WARN] could not get {video_path} under {SIZE_LIMIT_BYTES/1048576:.0f}MB; sending best effort")
    return dst if os.path.exists(dst) else video_path


def _video_data_url(path: str) -> str:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:video/mp4;base64,{b64}"


def make_call(model_name: str):
    def call(video_path: str, prompt: str) -> dict:
        try:
            send_path = compress_if_needed(video_path)
            data_url = _video_data_url(send_path)
        except Exception as e:
            print(f"[ERROR] prep failed for {video_path}: {e}")
            return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

        attempt = 0
        while True:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "video_url", "video_url": {"url": data_url}},
                        ],
                    }],
                )
                text = ""
                if resp.choices and resp.choices[0].message is not None:
                    text = resp.choices[0].message.content or ""
                if not text.strip():
                    attempt += 1
                    if attempt <= 8:
                        wait = min(2 ** attempt * 3, 60)
                        print(f"\n[RETRY {attempt}] empty/no-choices response, waiting {wait}s")
                        time.sleep(wait)
                        continue
                    print(f"\n[ERROR] persistent empty response for {video_path}")
                    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}
                pt = ct = 0
                if resp.usage:
                    pt = resp.usage.prompt_tokens or 0
                    ct = resp.usage.completion_tokens or 0
                return {"text": text, "prompt_tokens": pt, "completion_tokens": ct}
            except Exception as e:
                msg = str(e)
                if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "502", "rate", "overloaded")):
                    attempt += 1
                    wait = min(2 ** attempt * 5, 120)
                    print(f"\n[RETRY {attempt}] {msg[:80]}, waiting {wait}s")
                    time.sleep(wait)
                    if attempt <= 8:
                        continue
                print(f"\n[ERROR] OpenRouter call failed for {video_path}: {msg[:160]}")
                return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

    return call


def preflight():
    print("[PREFLIGHT] OpenRouter key + model ...")
    r = client.chat.completions.create(
        model=GEMINI_MODEL, max_tokens=10,
        messages=[{"role": "user", "content": "reply OK"}],
    )
    print(f"[PREFLIGHT] OpenRouter OK: {r.choices[0].message.content!r}")
    print(f"[PREFLIGHT] matcher at {SEMANTIC_MATCHER_URL}/health ...")
    resp = requests.get(f"{SEMANTIC_MATCHER_URL}/health", timeout=10)
    resp.raise_for_status()
    info = resp.json()
    print(f"[PREFLIGHT] matcher OK. backend={info.get('backend')}, model={info.get('model')}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a Gemini model through OpenRouter")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--titles", nargs="+", default=None, help="only these dataset folders (default: all)")
    p.add_argument("--max-examples", type=int, default=None, help="limit the number of clips")
    p.add_argument("--output-csv", default=None, help="results file (default: runs/<model>.csv)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    GEMINI_MODEL = args.model

    try:
        preflight()
    except Exception as e:
        print(f"[FATAL] Preflight failed: {e}")
        sys.exit(1)

    safe = args.model.replace("/", "_").replace(".", "_")
    run_evaluation(
        model_call_fn=make_call(args.model),
        model_name=args.model,
        dataset_dir=DATASET_DIR,
        output_csv=args.output_csv or os.path.join(OUTPUT_DIR, f"{safe}.csv"),
        conditions=args.conditions,
        titles=args.titles,
        max_examples=args.max_examples,
    )
