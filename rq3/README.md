# RQ3: why 37 buggy clips are still missed

Under GlitchGuard_s, no model detects 37 of the 135 buggy clips: no model flags the clip with a description the matcher accepts. Thirty of them are flagged by at least one model, but never with an accepted description; seven are never flagged. Two authors watched each clip and labeled it together using the codebook below.

## Files

| File | Content |
|---|---|
| `taxonomy_labels.csv` | one row per missed clip: the bug, the authors' answers to the codebook, the resulting category, and the model output behind the answers (columns below) |
| `detected_clips_coverage.csv` | for each of the 98 buggy clips that at least one model detects: the detecting models, the property IDs they cite, the checklist property that describes the bug, and whether a detecting model cites it |

`python3 scripts/reproduce_paper.py`, run from the package root, prints the counts below.

Columns of `taxonomy_labels.csv`:

| Column | Content |
|---|---|
| `item`, `file_id`, `game` | row number, clip, title |
| `summary` | the row's answers in one sentence, ending with the group used in the paper (description failure, perception failure, coverage gap or matcher error) |
| `gt_descr1`, `gt_descr2` | the two ground-truth descriptions of the bug (from the dataset) |
| `gt_clear` | codebook question 1: is the ground truth clear (`yes`/`no`) |
| `covering_ids`, `covering_names` | question 2: the checklist properties whose check question describes the bug (`none` if no property does) |
| `best_report` | question 3: `match`, `partial`, `judged_normal` or `missed` |
| `best_model` | question 4: the model behind the best report (empty for `missed`) |
| `best_model_output` | that model's GlitchGuard_s answer: its bug description for `match` and `partial`, its reasoning for `judged_normal` |
| `other_visible` | question 5: a model flags another anomaly that is visible in the clip but not in the ground truth |
| `category` | follows from the answers (rules below) |
| `accepted_in_other_conditions` | a description the matcher accepted for this clip in the baseline or GlitchGuard_g runs, with the condition and model |
| `note` | the authors' note where a decision needed one |

Columns of `detected_clips_coverage.csv`:

| Column | Content |
|---|---|
| `clip`, `game` | clip and title |
| `detecting_models` | the models that detect the clip under GlitchGuard_s (bug reported and description accepted by the matcher) |
| `cited_ids` | the game-specific property IDs that these models cite in their reasoning |
| `covering_property` | the checklist property whose check question describes the bug (empty if none) |
| `covered` | `yes` if such a property exists |
| `covering_property_cited` | `yes` if a detecting model cites that property |
| `note` | short explanation for the four clips no property covers and for judgment calls |
| `gt1`, `gt2` | the two ground-truth descriptions of the bug (from the dataset) |

## Counts

| Category | Clips | In the paper |
|---|---|---|
| MISDESCRIBED | 21 (18 partial, 3 match) | 18 description failures, 3 matcher errors |
| NOT_REPORTED | 10 (7 missed, 3 judged normal) | 10 perception failures |
| NOT_COVERED | 6 | 6 coverage gaps |
| AMBIGUOUS | 0 | every ground truth was judged clear |

Of the seven clips no model flags, five are NOT_REPORTED and two NOT_COVERED. Clip `ktrxq0` is NOT_COVERED although G-Pro gave a correct description that the matcher rejected; the paper counts it only as a coverage gap.

## Codebook

Watch the clip, then answer in order.

1. **Ground truth clear?** *No* if the anomaly the ground truth (GT) describes is not visible, if GT1 and GT2 describe different events and it is unclear which is the bug, or if the GT describes plausibly normal gameplay. GT1 and GT2 often describe one glitch from two angles, such as cause and effect; that is still *yes*. If GT2 is empty, use GT1.
2. **Covering properties.** Every property in the title's checklist, as the models saw it, whose check question describes the GT anomaly, including any cause the question requires (for example, RDR-007 requires a gunshot). *None* if nothing fits.
3. **Best model report.** Read all seven outputs, including the reasoning of models that answered "no glitch", and pick the first that applies:
   - *match*: a flagged description describes the GT anomaly, so the matcher should have accepted it;
   - *partial*: a flagged description reports the same event (the same object at the same moment, or the event's direct cause or effect) but a different or incomplete anomaly;
   - *judged_normal*: no flagged description reports the event, but some model's reasoning describes it and calls it normal;
   - *missed*: no model describes the event.
4. **Best model.** The model behind that report (empty for *missed*).
5. **Other anomaly visible?** *Yes* if a model flags a different anomaly that is visible in the clip but not in the GT.

| Category | Rule | Meaning |
|---|---|---|
| AMBIGUOUS | GT not clear | cannot count against the models |
| NOT_COVERED | GT clear, no covering property | coverage gap |
| NOT_REPORTED | covered; *judged_normal* or *missed* | perception failure (judgment failure for *judged_normal*) |
| MISDESCRIBED | covered; *match* or *partial* | the model saw the event but described it differently; *match* marks a matcher false negative |