import argparse
import base64
import os
import sys

import requests
from openai import OpenAI

from shared_runner import CONDITIONS, run_evaluation

_HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_BASE_URL = os.getenv("CLUSTER_VLM_SERVER", "http://127.0.0.1:8000/v1")
SEMANTIC_MATCHER_URL = os.getenv("SEMANTIC_MATCHER_URL", "http://127.0.0.1:8002")
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(_HERE, os.pardir, "dataset", "Games_json"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(_HERE, os.pardir, "runs"))

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
TARGET_FPS = 2.0
MAX_DIMENSION = 1280
JPEG_QUALITY = 95

CLIENT = OpenAI(base_url=SERVER_BASE_URL, api_key="NOT_USED", timeout=1800.0)


def extract_frames_base64(video_path: str, max_frames: int, letterbox_ultrawide: bool) -> list:
    """Return the sampled frames as OpenAI-style image_url content parts."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    interval = max(1, int(fps / TARGET_FPS))
    frame_indices = list(range(0, total_frames, interval))
    if len(frame_indices) > max_frames:
        step = len(frame_indices) / max_frames
        frame_indices = [frame_indices[int(i * step)] for i in range(max_frames)]

    frames_b64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        if letterbox_ultrawide and w * 9 > h * 16:
            pad = int(round(w * 9 / 16)) - h
            top = pad // 2
            frame = cv2.copyMakeBorder(frame, top, pad - top, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            h, w = frame.shape[:2]
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        frames_b64.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    cap.release()
    print(f"    [FRAMES] {os.path.basename(os.path.dirname(video_path))}: {len(frames_b64)} frames "
          f"from {duration:.1f}s video ({fps:.0f} fps source)")
    return frames_b64


def make_call_cluster(model_id: str, max_frames: int, letterbox_ultrawide: bool):
    def call_vlm(video_path: str, prompt: str) -> dict:
        frames = extract_frames_base64(video_path, max_frames, letterbox_ultrawide)
        if not frames:
            return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

        response = CLIENT.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}] + frames}],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
        )
        usage = response.usage
        return {
            "text": response.choices[0].message.content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }

    return call_vlm


def preflight_checks(model_id: str) -> None:
    print(f"[PREFLIGHT] vLLM server at {SERVER_BASE_URL} ...")
    try:
        resp = requests.get(f"{SERVER_BASE_URL}/models", timeout=30)
        resp.raise_for_status()
        available = [m["id"] for m in resp.json().get("data", [])]
        print(f"[PREFLIGHT] vLLM OK. Models: {available}")
        if model_id not in available:
            print(f"[WARN] Requested '{model_id}' not in {available}")
    except Exception as e:
        raise RuntimeError(f"Cannot reach the vLLM server at {SERVER_BASE_URL}: {e}") from e

    print(f"[PREFLIGHT] Matcher at {SEMANTIC_MATCHER_URL}/health ...")
    try:
        resp = requests.get(f"{SEMANTIC_MATCHER_URL}/health", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        print(f"[PREFLIGHT] Matcher OK. Backend: {info.get('backend')}, Model: {info.get('model')}")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach the matcher at {SEMANTIC_MATCHER_URL}. Start it with: "
            f"uvicorn server_semantic_matcher_claude:app --port 8002\nError: {e}"
        ) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an open-weight VLM served by vLLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model id on the server (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-frames", type=int, default=25, help="frame cap per clip (default: 25)")
    parser.add_argument("--letterbox-ultrawide", action="store_true",
                        help="pad frames wider than 16:9 to 16:9 (used for InternVL3)")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--titles", nargs="+", default=None,
                        help="only these dataset folders, e.g. Fallout_76 Watch_Dogs (default: all)")
    parser.add_argument("--max-examples", type=int, default=None, help="limit the number of clips")
    parser.add_argument("--output-csv", default=None, help="results file (default: runs/<model>.csv)")
    args = parser.parse_args()

    try:
        preflight_checks(args.model)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    safe_name = args.model.split("/")[-1].replace(".", "_")
    output_csv = args.output_csv or os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
    print(f"[INFO] Frames: {TARGET_FPS:g} fps, max {args.max_frames}, longer side {MAX_DIMENSION}px, "
          f"letterbox ultrawide: {args.letterbox_ultrawide}")
    run_evaluation(
        model_call_fn=make_call_cluster(args.model, args.max_frames, args.letterbox_ultrawide),
        model_name=args.model,
        dataset_dir=DATASET_DIR,
        output_csv=output_csv,
        conditions=args.conditions,
        titles=args.titles,
        max_examples=args.max_examples,
    )
