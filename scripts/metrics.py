from typing import Dict, Iterable, List


def as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1")


def outcome(row: Dict, cond: str) -> str:
    buggy = as_bool(row["gt_is_glitch"])
    flag = as_bool(row[f"{cond}_glitch"])
    match = as_bool(row[f"{cond}_correct_match"])
    if buggy:
        return ("TP" if match else "FN_wrong") if flag else "FN_missed"
    return "FP" if flag else "TN"


def compute(rows: List[Dict], cond: str) -> Dict[str, float]:
    n = {k: 0 for k in ("TP", "FN_wrong", "FN_missed", "FP", "TN")}
    for r in rows:
        if not as_bool(r.get("has_gt", True)):
            continue
        n[outcome(r, cond)] += 1
    tp, fp, tn = n["TP"], n["FP"], n["TN"]
    fn = n["FN_wrong"] + n["FN_missed"]
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp + n["FN_wrong"]) if tp + fp + n["FN_wrong"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(TP=tp, FN=fn, FN_wrong=n["FN_wrong"], FP=fp, TN=tn,
                recall=rec, precision=prec, f1=f1, fpr=fpr)


def print_metrics(rows: List[Dict], conditions: Iterable[str], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print(f"  {'condition':10} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} {'Rec.':>7} {'Prec.':>7} {'F1':>7} {'FPR':>7}")
    for c in conditions:
        m = compute(rows, c)
        print(f"  {c:10} {m['TP']:4d} {m['FN']:4d} {m['FP']:4d} {m['TN']:4d} "
              f"{100 * m['recall']:6.1f}% {100 * m['precision']:6.1f}% {100 * m['f1']:6.1f}% {100 * m['fpr']:6.1f}%")
