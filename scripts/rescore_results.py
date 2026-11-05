#!/usr/bin/env python3
import argparse
import csv

from metrics import as_bool, print_metrics
from shared_data import descriptions_match

CONDITIONS = ("baseline", "generic", "specific")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="results file (one row per clip)")
    ap.add_argument("--out", required=True, help="where to write the re-scored file")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    conds = [c for c in CONDITIONS if f"{c}_glitch" in fields]

    calls = changed = 0
    for r in rows:
        if not as_bool(r["has_gt"]):
            continue
        gt1, gt2 = r["gt_descr1"] or None, r["gt_descr2"] or None
        for c in conds:
            detect = as_bool(r[f"{c}_glitch"])
            if as_bool(r["gt_is_glitch"]):
                desc = r[f"{c}_desc"]
                correct = bool(descriptions_match(desc, gt1, gt2)) if detect and desc else False
                calls += detect and bool(desc)
            else:
                correct = not detect
            changed += str(correct) != r[f"{c}_correct_match"]
            r[f"{c}_correct_match"] = str(correct)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[INFO] {len(rows)} clips, {calls} matcher calls, {changed} verdicts differ from the input. Wrote {args.out}")
    print_metrics(rows, conds, title="Description-aware metrics after re-scoring")


if __name__ == "__main__":
    main()
