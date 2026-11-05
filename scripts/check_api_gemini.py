import argparse
import os
import sys
import time

import requests
from google import genai
from google.genai import types

from shared_runner import CONDITIONS, run_evaluation

_HERE = os.path.dirname(os.path.abspath(__file__))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. "
        "Get your API key from: https://aistudio.google.com/apikey"
    )

SEMANTIC_MATCHER_URL = os.getenv("SEMANTIC_MATCHER_URL", "http://127.0.0.1:8002")

DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(_HERE, os.pardir, "dataset", "Games_json"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(_HERE, os.pardir, "runs"))

GEMINI_MODEL = "gemini-2.5-pro"

client = genai.Client(api_key=GEMINI_API_KEY)


def make_call_gemini(model_name: str):
    """Create a call function bound to a specific Gemini model.

    Caches the most recently uploaded video so the conditions of one clip
    reuse a single upload.
    """
    cache = {"path": None, "file": None}

    def _get_uploaded(video_path: str):
        if cache["path"] == video_path and cache["file"] is not None:
            return cache["file"]
        if cache["file"] is not None:
            try:
                client.files.delete(name=cache["file"].name)
            except Exception:
                pass
            cache["path"], cache["file"] = None, None

        uploaded_file = client.files.upload(file=video_path)
        while uploaded_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        cache["path"], cache["file"] = video_path, uploaded_file
        return uploaded_file

    def call_gemini(video_path: str, prompt: str) -> dict:
        attempt = 0
        while True:
            try:
                uploaded_file = _get_uploaded(video_path)

                if uploaded_file.state.name == "FAILED":
                    print(f"\n[ERROR] Video processing failed for {video_path}")
                    cache["path"], cache["file"] = None, None
                    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

                response = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(temperature=0.0),
                )

                prompt_tokens = 0
                completion_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

                return {
                    "text": response.text or "",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            except Exception as e:
                error_str = str(e)
                if ("503" in error_str or "UNAVAILABLE" in error_str or
                    "429" in error_str or "RESOURCE_EXHAUSTED" in error_str):
                    attempt += 1
                    wait = min(2 ** attempt * 10, 300)
                    print(f"\n[RETRY {attempt}] {error_str[:80]}... waiting {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"\n[ERROR] Gemini call failed for {video_path}: {e}")
                    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

    return call_gemini


def preflight_checks() -> None:
    """Verify Gemini API and semantic matcher are reachable."""

    print(f"[PREFLIGHT] Testing Gemini API key ...")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents="Say hello in one short sentence."
        )
        print(f"[PREFLIGHT] Gemini OK. Reply: {response.text!r}")
    except Exception as e:
        raise RuntimeError(f"Gemini API test failed: {e}") from e

    print(f"[PREFLIGHT] Matcher at {SEMANTIC_MATCHER_URL}/health ...")
    resp = requests.get(f"{SEMANTIC_MATCHER_URL}/health", timeout=10)
    resp.raise_for_status()
    info = resp.json()
    print(f"[PREFLIGHT] Matcher OK. Backend: {info.get('backend')}, Model: {info.get('model')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Gemini model through the Gemini API")
    parser.add_argument("--model", default=GEMINI_MODEL, help=f"Gemini model name (default: {GEMINI_MODEL})")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--titles", nargs="+", default=None, help="only these dataset folders (default: all)")
    parser.add_argument("--max-examples", type=int, default=None, help="limit the number of clips")
    parser.add_argument("--output-csv", default=None, help="results file (default: runs/<model>.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    GEMINI_MODEL = args.model

    try:
        preflight_checks()
    except Exception as e:
        print(f"[FATAL] Preflight failed: {e}")
        sys.exit(1)

    safe_name = args.model.replace("/", "_").replace(".", "_")
    run_evaluation(
        model_call_fn=make_call_gemini(args.model),
        model_name=args.model,
        dataset_dir=DATASET_DIR,
        output_csv=args.output_csv or os.path.join(OUTPUT_DIR, f"{safe_name}.csv"),
        conditions=args.conditions,
        titles=args.titles,
        max_examples=args.max_examples,
    )
