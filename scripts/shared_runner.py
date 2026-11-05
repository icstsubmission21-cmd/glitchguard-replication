import csv
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from metrics import print_metrics
from shared_data import descriptions_match, extract_detection, iter_examples
from shared_prompts import BUILDERS

CONDITIONS = ("baseline", "generic", "specific")
GT_FIELDS = ["file_id", "game", "video_path", "gt_is_glitch", "has_gt", "gt_descr1", "gt_descr2"]
SUFFIXES = ["raw", "glitch", "desc", "valid_json", "correct_match", "time_s",
            "prompt_tokens", "completion_tokens"]


def _score(ex, text: str, has_gt: bool, is_glitch: Optional[bool]):
    """Return (bug reported, description, valid JSON, correct match) for one answer."""
    detect, desc, valid = extract_detection(text)
    correct_match = False
    if has_gt:
        if is_glitch:
            if detect:
                correct_match = descriptions_match(desc, ex.gt_descr1, ex.gt_descr2)
        else:
            correct_match = not detect
    return detect, desc, valid, correct_match


def _title_folder(ex) -> str:
    return os.path.basename(os.path.dirname(ex.clip_dir.rstrip(os.sep)))


def _write(rows: List[Dict[str, Any]], fields: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    os.replace(tmp, path)


def run_evaluation(
    *,
    model_call_fn: Callable[[str, str], Dict[str, Any]],
    model_name: str,
    dataset_dir: str,
    output_csv: str,
    conditions: Iterable[str] = CONDITIONS,
    titles: Optional[Iterable[str]] = None,
    max_examples: Optional[int] = None,
) -> None:
    """Evaluate every clip of dataset_dir (optionally only some title folders) in the given conditions.

    model_call_fn(video_path, prompt) must return {"text", "prompt_tokens", "completion_tokens"}.
    """
    conditions = [c for c in CONDITIONS if c in set(conditions)]
    fields = GT_FIELDS + [f"{c}_{s}" for c in conditions for s in SUFFIXES]

    examples = list(iter_examples(dataset_dir))
    if titles:
        keep = set(titles)
        examples = [ex for ex in examples if _title_folder(ex) in keep]
    if max_examples is not None:
        examples = examples[:max_examples]
    if not examples:
        print("[FATAL] No clips found. Check the dataset directory and --titles.")
        return
    print(f"[INFO] {len(examples)} clips x {len(conditions)} conditions ({', '.join(conditions)}), model {model_name}\n")

    rows: List[Dict[str, Any]] = []
    if os.path.exists(output_csv):
        with open(output_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"[RESUME] {len(rows)} clips already in {output_csv}; continuing with the rest.")
    done = {r["video_path"] for r in rows}
    root = os.path.dirname(os.path.abspath(dataset_dir).rstrip(os.sep))

    for idx, ex in enumerate(examples, start=1):
        rel_path = os.path.relpath(os.path.abspath(ex.video_path), root).replace(os.sep, "/")
        if rel_path in done:
            continue
        print(f"=== {idx}/{len(examples)}: {ex.file_id} | {ex.game} ===")
        has_gt = ex.gt_is_glitch is not None
        row: Dict[str, Any] = {
            "file_id": ex.file_id,
            "game": ex.game,
            "video_path": rel_path,
            "gt_is_glitch": ex.gt_is_glitch,
            "has_gt": has_gt,
            "gt_descr1": ex.gt_descr1,
            "gt_descr2": ex.gt_descr2,
        }
        for cond in conditions:
            prompt = BUILDERS[cond](ex.game)
            t0 = time.time()
            try:
                result = model_call_fn(ex.video_path, prompt)
            except Exception as e:
                print(f"[ERROR] {cond} call failed for {ex.file_id}: {e}")
                result = {"text": "", "prompt_tokens": 0, "completion_tokens": 0}
            elapsed = time.time() - t0
            detect, desc, valid, match = _score(ex, result.get("text", ""), has_gt, ex.gt_is_glitch)
            print(f"    [{cond.upper():8}] glitch={detect}, valid_json={valid}, desc={desc!r} ({elapsed:.1f}s)")
            row.update({
                f"{cond}_raw": result.get("text", ""),
                f"{cond}_glitch": detect,
                f"{cond}_desc": desc,
                f"{cond}_valid_json": valid,
                f"{cond}_correct_match": match,
                f"{cond}_time_s": round(elapsed, 2),
                f"{cond}_prompt_tokens": result.get("prompt_tokens", 0),
                f"{cond}_completion_tokens": result.get("completion_tokens", 0),
            })
        rows.append(row)
        _write(rows, fields, output_csv)

    print(f"\n[INFO] Results: {output_csv}")
    print_metrics(rows, conditions, title=f"{model_name}: description-aware metrics")
