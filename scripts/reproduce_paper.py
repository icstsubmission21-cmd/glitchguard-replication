#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("G3-Pro", "gemini-3.1-pro"),
    ("G-Pro", "gemini-2.5-pro"),
    ("G-Flash", "gemini-2.5-flash"),
    ("I-78B", "internvl3-78b"),
    ("I-8B", "internvl3-8b"),
    ("Q-32B", "qwen2.5-vl-32b"),
    ("Q-7B", "qwen2.5-vl-7b"),
]
ORDER = [m for m, _ in MODELS]
GEMINI = ["G3-Pro", "G-Pro", "G-Flash"]
OPEN = ["I-78B", "I-8B", "Q-32B", "Q-7B"]
CONDS = ["baseline", "generic", "specific"]
COND_LABEL = {"baseline": "Baseline", "generic": "GlitchGuard_g", "specific": "GlitchGuard_s"}

FOLDER_GAME = {
    "Cyberpunk_2077_Dataset": "Cyberpunk 2077",
    "Fallout_4": "Fallout 4",
    "Far_Cry_5_Dataset": "Far Cry 5",
    "GTAV_Dataset": "GTA V",
    "Just_Cause_3": "Just Cause 3",
    "RDR2_Dataset": "RDR2",
    "The_Elder_scrolls_V_Skyrim": "Skyrim",
    "The_Witcher_3_Dataset": "Witcher 3",
    "Fallout_76": "Fallout 76",
    "Watch_Dogs": "Watch Dogs 2",
}
GAMES = list(FOLDER_GAME.values())

SPECIFIC_ID = re.compile(r"\b(?:CP|FO|F76|FC|GTA|JC|RDR|SK|W3|WD2)-\d{3}\b", re.I)
GENERIC_ID = re.compile(r"\b(?:MR|GP)-?0*(\d{1,3})\b", re.I)

PRICE = {"G-Flash": (0.30, 2.50), "G-Pro": (1.25, 10.00)}

TABLES = {}

CATEGORY_LABEL = {
    "character_physics": "Character physics", "vehicle": "Vehicle", "collision": "Collision",
    "character_animation": "Character animation", "environment_physics": "Environment physics",
    "character_model": "Character model", "camera": "Camera", "ui": "UI", "rendering": "Rendering",
    "projectile": "Projectile", "object": "Object", "audio": "Audio",
}


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def r1(x):
    """Round half up to one decimal (as in the paper)."""
    return math.floor(x * 10 + 0.5 + 1e-9) / 10


def r0(x):
    return int(math.floor(x + 0.5 + 1e-9))


def f1s(x):
    return f"{r1(x):.1f}"


def boolean(v):
    v = v.strip()
    if v not in ("True", "False"):
        raise ValueError(f"unexpected boolean value {v!r}")
    return v == "True"


def reasoning_of(raw):
    try:
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1]).get("reasoning", "") or ""
    except (ValueError, json.JSONDecodeError, AttributeError):
        return ""


def specific_ids(text):
    return sorted({x.upper() for x in SPECIFIC_ID.findall(text)})


def show(title, header, rows, note=None):
    TABLES[title] = (header, rows)
    print()
    print(title)
    print("-" * len(title))
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    if note:
        print(note)


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_results():
    data = {}
    for model, stem in MODELS:
        path = ROOT / "results" / f"{stem}.csv"
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 270:
            sys.exit(f"{path}: expected 270 clips, found {len(rows)}")
        for cond in CONDS:
            out = {}
            for r in rows:
                _, folder, clip, _ = r["video_path"].split("/")
                key = (FOLDER_GAME[folder], clip)
                buggy = boolean(r["gt_is_glitch"])
                flag = boolean(r[f"{cond}_glitch"])
                match = boolean(r[f"{cond}_correct_match"])
                raw = r[f"{cond}_raw"]
                if buggy:
                    state = ("TP" if match else "W") if flag else "N"
                else:
                    state = "FP" if flag else "TN"
                out[key] = dict(
                    game=key[0], file_id=key[1], buggy=buggy, flag=flag, state=state,
                    usable=bool(raw.strip()) and boolean(r[f"{cond}_valid_json"]),
                    raw=raw, reasoning=reasoning_of(raw), desc=r[f"{cond}_desc"],
                    prompt_tokens=int(r[f"{cond}_prompt_tokens"]) if f"{cond}_prompt_tokens" in r else None,
                    completion_tokens=int(r[f"{cond}_completion_tokens"]) if f"{cond}_completion_tokens" in r else None,
                )
            data[model, cond] = out
    keys = set(data["G3-Pro", "baseline"])
    for d in data.values():
        assert set(d) == keys
    return data


def load_properties():
    """properties/generic.json and the game-specific files properties/<title>.json."""
    generic = json.loads((ROOT / "properties" / "generic.json").read_text(encoding="utf-8"))
    titles = {p.stem: json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((ROOT / "properties").glob("*.json")) if p.name != "generic.json"}
    return generic, titles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="directory for CSV copies of the tables")
    args = ap.parse_args()

    D = load_results()
    generic, titles = load_properties()
    names = {r["id"]: r["name"] for j in [generic, *titles.values()] for r in j["properties"]}
    generic_cat = {r["id"]: r["category"] for r in generic["properties"]}
    keys = sorted(D["G3-Pro", "baseline"])
    bug = [k for k in keys if D["G3-Pro", "baseline"][k]["buggy"]]
    clean = [k for k in keys if k not in set(bug)]
    print(f"Clips: {len(keys)} ({len(bug)} buggy, {len(clean)} clean); models: {len(ORDER)}")

    cat_rows = []
    clips_by_game = Counter(k[0] for k in bug)
    title_label = {"cyberpunk_2077": "Cyberpunk 2077", "watch_dogs_2": "Watch Dogs 2", "fallout_76": "Fallout 76",
                   "red_dead_redemption_2": "RDR2", "fallout_4": "Fallout 4", "grand_theft_auto_v": "GTA V",
                   "far_cry_5": "Far Cry 5", "the_witcher_3": "Witcher 3", "skyrim": "Skyrim",
                   "just_cause_3": "Just Cause 3"}
    assert sorted(titles) == sorted(title_label)
    total_props = 0
    for stem, j in titles.items():
        n = len(j["properties"])
        total_props += n
        cat_rows.append([title_label[stem], j.get("engine", ""), clips_by_game[title_label[stem]], n])
    cat_rows.sort(key=lambda r: (-r[3], -r[2], r[0]))
    gen_counts = Counter(r["category"] for r in generic["properties"])
    show("Catalog: game-specific properties and buggy clips per title (Table II b)",
         ["Game", "Engine", "Clips", "#Properties"], cat_rows + [["Total", "", len(bug), total_props]])
    show("Catalog: generic properties per category (Table II a)", ["Category", "#"],
         [[CATEGORY_LABEL.get(c, c), n] for c, n in sorted(gen_counts.items(), key=lambda t: (-t[1], t[0]))] +
         [["Total", sum(gen_counts.values())]])

    M = {}
    for m in ORDER:
        for c in CONDS:
            n = Counter(D[m, c][k]["state"] for k in keys)
            tp, w, nn, fp, tn = n["TP"], n["W"], n["N"], n["FP"], n["TN"]
            rec, prec = pct(tp, tp + w + nn), pct(tp, tp + fp + w)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            M[m, c] = dict(TP=tp, FN=w + nn, W=w, N=nn, FP=fp, TN=tn, reports=tp + fp + w,
                           rec=rec, prec=prec, f1=f1, fpr=pct(fp, fp + tn))
    rows = []
    for m in ORDER:
        b = M[m, "baseline"]
        for c in CONDS:
            x = M[m, c]
            rows.append([m, COND_LABEL[c], x["TP"], x["FN"], x["FP"], x["TN"],
                         f1s(x["rec"]), f1s(x["prec"]), f1s(x["f1"]), f1s(x["fpr"])])
            if c != "baseline":
                rows.append(["", "  delta vs baseline", x["TP"] - b["TP"], x["FN"] - b["FN"], x["FP"] - b["FP"],
                             x["TN"] - b["TN"], f1s(x["rec"] - b["rec"]), f1s(x["prec"] - b["prec"]),
                             f1s(x["f1"] - b["f1"]), f1s(x["fpr"] - b["fpr"])])
    show("Main results (Table IV); rates in %, deltas in counts and percentage points",
         ["Model", "Approach", "TP", "FN", "FP", "TN", "Rec.", "Prec.", "F1", "FPR"], rows)

    trans = {}
    for m in ORDER:
        before, after = D[m, "baseline"], D[m, "specific"]
        improved = [k for k in bug if before[k]["state"] != "TP" and after[k]["state"] == "TP"]
        regressed = [k for k in bug if before[k]["state"] == "TP" and after[k]["state"] != "TP"]
        stayed_tp = [k for k in bug if before[k]["state"] == "TP" and after[k]["state"] == "TP"]
        stayed_fn = [k for k in bug if before[k]["state"] != "TP" and after[k]["state"] != "TP"]
        cite = [k for k in improved if specific_ids(after[k]["reasoning"])]
        trans[m] = dict(improved=improved, regressed=regressed, stayed_tp=stayed_tp, stayed_fn=stayed_fn, cite=cite)
    show("McNemar's exact test (two-sided), baseline vs GlitchGuard_s, description-aware TP on the buggy clips",
         ["Model", "b (improved)", "c (regressed)", "p"],
         [[m, len(trans[m]["improved"]), len(trans[m]["regressed"]),
           f"{mcnemar_exact(len(trans[m]['improved']), len(trans[m]['regressed'])):.4f}"] for m in ORDER])
    ups = sum(len(trans[m]["improved"]) > len(trans[m]["regressed"]) for m in ORDER)
    downs = sum(len(trans[m]["improved"]) < len(trans[m]["regressed"]) for m in ORDER)
    print(f"\nSign test across models: recall up for {ups} of {len(ORDER)} models, down for {downs}; "
          f"p = {mcnemar_exact(ups, downs):.4f} (exact, two-sided)")

    rows = [
        ["Stayed TP"] + [len(trans[m]["stayed_tp"]) for m in ORDER],
        ["Improved"] + [len(trans[m]["improved"]) for m in ORDER],
        ["Regressed"] + [len(trans[m]["regressed"]) for m in ORDER],
        ["Stayed FN"] + [len(trans[m]["stayed_fn"]) for m in ORDER],
        ["Net"] + [f"+{len(trans[m]['improved']) - len(trans[m]['regressed'])}" for m in ORDER],
        ["Cite ID"] + [len(trans[m]["cite"]) for m in ORDER],
        ["Cite rate"] + [f"{r0(pct(len(trans[m]['cite']), len(trans[m]['improved'])))}%" for m in ORDER],
        ["Missed entirely"] + [M[m, "specific"]["N"] for m in ORDER],
        ["Wrong description"] + [M[m, "specific"]["W"] for m in ORDER],
        ["Total FN"] + [M[m, "specific"]["FN"] for m in ORDER],
    ]
    show("Outcome shifts under GlitchGuard_s, property-ID citations in improved clips, FN sources (Table V)",
         [""] + ORDER, rows,
         note="Cite ID: improved clips whose reasoning names at least one game-specific property ID.")

    rows = []
    for g in GAMES:
        gk = [k for k in bug if k[0] == g]
        row = [g, len(gk)]
        for m in ORDER:
            for c in ["baseline", "specific"]:
                row.append(f1s(pct(sum(D[m, c][k]["state"] == "TP" for k in gk), len(gk))))
        rows.append(row)
    show("Per-game recall (%), Base vs +Prop (GlitchGuard_s) (Table VI)",
         ["Game", "n"] + [f"{m} {c}" for m in ORDER for c in ["Base", "+Prop"]], rows)

    cross = defaultdict(Counter)
    for m in ORDER:
        for k in trans[m]["improved"]:
            for i in specific_ids(D[m, "specific"][k]["reasoning"]):
                cross[i][m] += 1
    rows = [[i, names.get(i, ""), len(c), sum(c.values())] for i, c in cross.items() if len(c) >= 4]
    rows.sort(key=lambda r: (-r[2], -r[3], r[0]))
    show("Properties cited by four or more models in improved clips (Table VII)",
         ["Property ID", "Name (checklist file)", "#Models", "Total"], rows)

    credit = defaultdict(Counter)
    pairs, cited_pairs = Counter(), Counter()
    for m in ORDER:
        for k in keys:
            r = D[m, "generic"][k]
            if r["state"] not in ("TP", "FP"):
                continue
            ids = {f"MR-{int(v):03d}" for v in GENERIC_ID.findall(r["reasoning"])} & set(generic_cat)
            cats = {generic_cat[i] for i in ids}
            pairs[r["state"]] += 1
            cited_pairs[r["state"]] += bool(cats)
            for c in cats:
                credit[c][r["state"]] += 1
    sizes = Counter(generic_cat.values())
    rows = []
    for c in sorted(sizes, key=lambda c: (-credit[c]["TP"], -credit[c]["FP"], c)):
        rows.append([CATEGORY_LABEL.get(c, c), sizes[c], credit[c]["TP"], credit[c]["FP"],
                     f"{credit[c]['TP'] / sizes[c]:.1f}"])
    det_total = sum(credit[c]["TP"] for c in sizes)
    fa_total = sum(credit[c]["FP"] for c in sizes)
    rows.append(["Total", sum(sizes.values()), det_total, fa_total, ""])
    show("Per-category effectiveness of the generic properties, GlitchGuard_g runs, all seven models (Table VIII)",
         ["Category", "#Prop", "Det.", "FA", "D/P"], rows,
         note=(f"Detections (TP model-clip pairs): {pairs['TP']}; false alarms (flagged clean clips): {pairs['FP']}. "
               f"TP pairs citing a parseable generic property: {cited_pairs['TP']} "
               f"({r0(pct(cited_pairs['TP'], pairs['TP']))}%)."))
    cp, veh, col = credit["character_physics"]["TP"], credit["vehicle"]["TP"], credit["collision"]["TP"]
    print(f"Character physics + vehicle: {r0(pct(cp + veh, det_total))}% of detection citations; "
          f"+ collision: {r0(pct(cp + veh + col, det_total))}%. Collision: {r0(pct(col, det_total))}% of detection "
          f"citations, {credit['collision']['FP']} of {fa_total} false-alarm citations "
          f"({r0(pct(credit['collision']['FP'], fa_total))}%). Character animation: "
          f"{credit['character_animation']['FP']} false alarms vs {credit['character_animation']['TP']} detections.")

    val = {}
    for a in ["author1", "author2"]:
        with open(ROOT / "matcher_validation" / f"matcher_sample-{a}.csv", newline="", encoding="utf-8") as f:
            val[a] = {(r["game"], r["file_id"], r["model"], r["condition"]): r for r in csv.DictReader(f)}
    assert val["author1"].keys() == val["author2"].keys()
    vk = sorted(val["author1"])
    population = sum(D[m, c][k]["flag"] for m in ORDER for c in CONDS for k in bug)
    n0 = math.ceil(1.96 ** 2 * 0.25 / 0.05 ** 2)
    cochran = math.ceil(n0 / (1 + (n0 - 1) / population))
    rows = []
    for label, sel in [("All seven models", vk), ("G3-Pro only", [k for k in vk if k[2] == "G3-Pro"])]:
        h1 = [val["author1"][k]["human_label_author"] == "MATCH" for k in sel]
        h2 = [val["author2"][k]["human_label_author"] == "MATCH" for k in sel]
        mj = [val["author1"][k]["matcher_label"] == "MATCH" for k in sel]
        rows.append([label, len(sel), f"{kappa(h1, h2):.2f}", f"{kappa(mj, h1):.2f}", f"{kappa(mj, h2):.2f}",
                     sum(x != y for x, y in zip(mj, h1)), sum(x and not y for x, y in zip(h1, mj))])
    show("Matcher validation: Cohen's kappa (Section IV-B)",
         ["Sample", "n", "author1-author2", "matcher-author1", "matcher-author2",
          "matcher/author1 disagreements", "of which rejected by matcher"], rows,
         note=(f"Matcher decisions in all runs (flagged buggy clips, 7 models x 3 conditions): N = {population}; "
               f"Cochran (95% confidence, +/-5% margin): n0 = {n0}, n = ceil(n0 / (1 + (n0 - 1) / N)) = {cochran}; sample size {len(vk)}."))

    union_tp = [k for k in bug if any(D[m, "specific"][k]["state"] == "TP" for m in ORDER)]
    union_flag = [k for k in bug if any(D[m, "specific"][k]["flag"] and D[m, "specific"][k]["usable"] for m in ORDER)]
    rows = []
    for m in ORDER:
        miss = [k for k in bug if D[m, "specific"][k]["state"] != "TP"]
        other = [k for k in miss if any(D[o, "specific"][k]["state"] == "TP" for o in ORDER if o != m)]
        x = M[m, "specific"]
        rows.append([m, x["N"], x["W"], x["FN"], f"{r0(pct(x['W'], x['FN']))}%",
                     f"{r0(pct(x['N'], x['FN']))}%", len(miss), len(other), f"{r0(pct(len(other), len(miss)))}%"])
    show("RQ3: false negatives under GlitchGuard_s and cross-model recovery",
         ["Model", "Missed", "Wrong desc.", "FN", "Wrong desc. share", "Perception share",
          "Misses", "Detected by another model", "Share"], rows)
    print(f"Buggy clips correctly detected by at least one model under GlitchGuard_s: {len(union_tp)} of {len(bug)} "
          f"({r0(pct(len(union_tp), len(bug)))}%); flagged by at least one model: {len(union_flag)} "
          f"({r0(pct(len(union_flag), len(bug)))}%); never flagged: {len(bug) - len(union_flag)}.")
    with open(ROOT / "rq3" / "detected_clips_coverage.csv", newline="", encoding="utf-8") as f:
        cov = list(csv.DictReader(f))
    print(f"Coverage of the {len(cov)} detected clips (rq3/detected_clips_coverage.csv): a checklist property "
          f"describes the bug in {sum(r['covered'] == 'yes' for r in cov)}; a detecting model cites it in "
          f"{sum(r['covering_property_cited'] == 'yes' for r in cov)}.")
    det_ids = {k[1] for k in union_tp}
    assert {r["clip"] for r in cov} == det_ids, "coverage table and detected clips differ"
    with open(ROOT / "rq3" / "taxonomy_labels.csv", newline="", encoding="utf-8") as f:
        tax = list(csv.DictReader(f))
    tc = Counter((r["category"], r["best_report"]) for r in tax)
    missed_ids = {k[1] for k in bug if k not in set(union_tp)}
    assert {r["file_id"] for r in tax} == missed_ids, "taxonomy clips differ from the clips no model detects"
    print(f"Taxonomy of the {len(tax)} clips no model detects (rq3/taxonomy_labels.csv): "
          f"{tc['MISDESCRIBED', 'partial']} description failures, "
          f"{sum(v for (c, _), v in tc.items() if c == 'NOT_REPORTED')} perception failures, "
          f"{sum(v for (c, _), v in tc.items() if c == 'NOT_COVERED')} coverage gaps, "
          f"{tc['MISDESCRIBED', 'match']} matcher errors.")
    missed_key = {k[1]: k for k in bug if k not in set(union_tp)}
    never = [r for r in tax if not any(D[m, "specific"][missed_key[r["file_id"]]]["flag"] for m in ORDER)]
    nc = Counter(r["category"] for r in never)
    print(f"  Perception failures: {tc['NOT_REPORTED', 'missed']} missed, {tc['NOT_REPORTED', 'judged_normal']} judged "
          f"normal. Never flagged by any model: {len(never)} clips ({nc['NOT_REPORTED']} perception failures, "
          f"{nc['NOT_COVERED']} coverage gaps).")

    with open(ROOT / "annotations" / "gpro_rdr2_gtav_regressions.csv", newline="", encoding="utf-8") as f:
        ann = {(r["game"], r["file_id"]): r for r in csv.DictReader(f)}
    reg = [k for k in trans["G-Pro"]["regressed"] if k[0] in ("RDR2", "GTA V")]
    assert set(reg) == set(ann), "annotated clips differ from the G-Pro regressions on RDR2 and GTA V"
    rows = []
    for k in sorted(reg):
        s = D["G-Pro", "specific"][k]
        rows.append([k[0], k[1], s["state"], ann[k]["reason"], ";".join(specific_ids(s["reasoning"]))])
        assert (s["state"] == "N") == (ann[k]["reason"] == "dismissed as normal behavior")
    show("Discussion: G-Pro regressions on RDR2 and GTA V under GlitchGuard_s (reason: manual reading)",
         ["Game", "Clip", "State", "Reason", "IDs cited"], rows,
         note=str(Counter(r[3] for r in rows)))
    rows = []
    for g, fid in [("Cyberpunk 2077", "8qd9j1"), ("Far Cry 5", "0dbmmg"), ("Skyrim", "5et06l"),
                   ("Just Cause 3", "g3vn3b")]:
        s = D["G-Pro", "specific"][g, fid]
        rows.append([fid, g, "buggy" if s["buggy"] else "clean", s["state"], s["reasoning"][:110] + "..."])
    show("Discussion: reasoning examples, G-Pro under GlitchGuard_s (Table IX)",
         ["Clip", "Game", "GT", "State", "Reasoning (start)"], rows)
    rows = []
    for fid in ["kae98l", "kbqzb0", "kcsbg6", "kr5egn", "kvver3"]:
        k = ("Cyberpunk 2077", fid)
        hits = [m for m in ORDER if D[m, "specific"][k]["state"] == "TP" and "CP-001" in specific_ids(D[m, "specific"][k]["reasoning"])]
        rows.append([fid, "buggy" if D["G3-Pro", "baseline"][k]["buggy"] else "clean", ", ".join(hits)])
    show("Discussion: Cyberpunk 2077 clips detected with property CP-001 (Fig. 3)",
         ["Clip", "GT", "Models detecting it while citing CP-001"], rows)

    def cost(m, c):
        pin, pout = PRICE[m]
        i = sum(D[m, c][k]["prompt_tokens"] for k in keys) / len(keys)
        o = sum(D[m, c][k]["completion_tokens"] for k in keys) / len(keys)
        return (i * pin + o * pout) / 1e6, i, o
    cf, fin, _ = cost("G-Flash", "specific")
    cp_, _, _ = cost("G-Pro", "baseline")
    _, fbase, _ = cost("G-Flash", "baseline")
    print(f"\nCost per clip from logged input and output tokens at published prices: G-Flash with GlitchGuard_s "
          f"${cf:.4f} vs unguided G-Pro ${cp_:.4f} ({cp_ / cf:.1f}x); the checklist adds {fin - fbase:.0f} input "
          f"tokens per clip on average (G-Flash).")

    base_tp = [M[m, "baseline"]["TP"] for m in ORDER]
    print("\nNumbers quoted in the text")
    print(f"  Baseline: models fail to correctly identify {f1s(100 - pct(sum(base_tp), 135 * len(ORDER)))}% of bugs on "
          f"average; the strongest (G3-Pro) fails on {f1s(100 - M['G3-Pro', 'baseline']['rec'])}%.")
    gs = [M[m, "specific"]["rec"] - M[m, "baseline"]["rec"] for m in ORDER]
    gg = [M[m, "generic"]["rec"] - M[m, "baseline"]["rec"] for m in OPEN]
    pr = [M[m, "specific"]["prec"] for m in ORDER]
    print(f"  GlitchGuard_s recall gain: +{f1s(min(gs))} to +{f1s(max(gs))} pp on all models; precision "
          f"{f1s(min(pr))}-{f1s(max(pr))}%, higher than baseline on "
          f"{sum(M[m, 'specific']['prec'] > M[m, 'baseline']['prec'] for m in ORDER)} of 7 models.")
    print(f"  GlitchGuard_g recall gain on open-weight models: +{f1s(min(gg))} to +{f1s(max(gg))} pp; FPR higher "
          f"than baseline for {sum(M[m, 'generic']['fpr'] > M[m, 'baseline']['fpr'] for m in ORDER)} of 7 models.")
    wb = [pct(M[m, 'baseline']['W'], M[m, 'baseline']['W'] + M[m, 'baseline']['TP']) for m in ORDER]
    ws = [pct(M[m, 'specific']['W'], M[m, 'specific']['W'] + M[m, 'specific']['TP']) for m in ORDER]
    print(f"  Flagged buggy clips with a wrong description: baseline {r0(min(wb))}-{r0(max(wb))}%, "
          f"GlitchGuard_s {r0(min(ws))}-{r0(max(ws))}%.")
    fl_b, fl_s = M["G-Flash", "baseline"], M["G-Flash", "specific"]
    print(f"  G-Flash incorrect reports: {f1s(pct(fl_b['reports'] - fl_b['TP'], fl_b['reports']))}% -> "
          f"{f1s(pct(fl_s['reports'] - fl_s['TP'], fl_s['reports']))}%.")
    print(f"  Open-weight baseline reports per model: {min(M[m, 'baseline']['reports'] for m in OPEN)}-"
          f"{max(M[m, 'baseline']['reports'] for m in OPEN)}; I-78B reports {M['I-78B', 'baseline']['reports']} -> "
          f"{M['I-78B', 'specific']['reports']} (TP {M['I-78B', 'baseline']['TP']} -> {M['I-78B', 'specific']['TP']}); "
          f"Q-7B GlitchGuard_s {M['Q-7B', 'specific']['TP']} of {M['Q-7B', 'specific']['reports']} reports correct.")
    g3 = trans["G3-Pro"]["improved"]
    print(f"  G3-Pro: {sum(D['G3-Pro', 'baseline'][k]['state'] == 'W' for k in g3)} of its {len(g3)} recoveries "
          f"change a wrong description to a matched one.")
    f76 = [k for k in bug if k[0] == "Fallout 76"]
    reg76 = [k for k in f76 if k in set(trans["G3-Pro"]["regressed"])]
    imp76 = [k for k in f76 if k in set(trans["G3-Pro"]["improved"])]
    print(f"  G3-Pro on Fallout 76: {len(reg76)} regressed clips "
          f"({sum(D['G3-Pro', 'specific'][k]['state'] == 'W' for k in reg76)} wrong description, "
          f"{sum(D['G3-Pro', 'specific'][k]['state'] == 'N' for k in reg76)} no glitch), {len(imp76)} improved.")
    up = down = 0
    for g in GAMES:
        gk = [k for k in bug if k[0] == g]
        d = sum(D[m, "generic"][k]["state"] == "TP" for m in OPEN for k in gk) - \
            sum(D[m, "baseline"][k]["state"] == "TP" for m in OPEN for k in gk)
        up += d > 0
        down += d < 0
    print(f"  GlitchGuard_g pooled over the four open-weight models: recall up on {up} of 10 games, down on {down}.")
    q7 = sum(any(D["Q-7B", "specific"][k]["state"] == "TP" for k in bug if k[0] == g) for g in GAMES)
    print(f"  Q-7B with GlitchGuard_s: positive recall on {q7} of 10 games.")
    rates = [pct(len(trans[m]["cite"]), len(trans[m]["improved"])) for m in ORDER]
    print(f"  Improved clips citing a game-specific property ID: {r0(min(rates))}-{r0(max(rates))}%.")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for i, (title, (header, rows)) in enumerate(TABLES.items(), 1):
            name = re.sub(r"[^A-Za-z0-9]+", "_", title.split(";")[0].split("(")[0]).strip("_").lower()
            with open(out / f"{i:02d}_{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
        print(f"\nWrote {len(TABLES)} tables to {out}")


if __name__ == "__main__":
    main()
