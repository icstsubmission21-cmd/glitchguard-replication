import json
import os
from typing import Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
PROPERTIES_DIR = os.getenv("PROPERTIES_DIR", os.path.join(_HERE, os.pardir, "properties"))
GENERIC_FILE = "generic.json"

GAME_TO_SPECIFIC_FILE: Dict[str, str] = {
    "Red Dead Redemption 2": "red_dead_redemption_2.json",
    "Grand Theft Auto V": "grand_theft_auto_v.json",
    "Cyberpunk 2077": "cyberpunk_2077.json",
    "Far Cry 5": "far_cry_5.json",
    "Just Cause 3": "just_cause_3.json",
    "Fallout 4": "fallout_4.json",
    "The Elder Scrolls V - Skyrim": "skyrim.json",
    "The Witcher 3 - Wild Hunt": "the_witcher_3.json",
    "Fallout 76": "fallout_76.json",
    "Watch Dogs 2": "watch_dogs_2.json",
    "Far Cry": "far_cry_5.json",
    "Fallout": "fallout_4.json",
    "fallout 4": "fallout_4.json",
    "GTA V": "grand_theft_auto_v.json",
    "Just Cause": "just_cause_3.json",
    "The Elder Scrolls": "skyrim.json",
    "The Witcher 3": "the_witcher_3.json",
    "Watch Dogs": "watch_dogs_2.json",
}


def _checklist_from_path(path: str, checklist_only: bool) -> str:
    """Render the checklist of one properties file as prompt text.

    checklist_only: keep only the properties marked "in_checklist": true (game-specific files).
    """
    if not os.path.exists(path):
        print(f"[WARN] Properties file does not exist: {path}")
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load properties file {path}: {e}")
        return ""

    entries = spec.get("properties", [])
    if checklist_only:
        entries = [r for r in entries if r.get("in_checklist")]
    return "\n".join(f"  {r['id']} ({r['name']}): {r['check']}" for r in entries)


BASE_PROMPT_TEMPLATE = """You are a helpful assistant analyzing short video clips from the video game "{game}".

You may or may not see a bug or glitch in the clip (for example physics violations, animation problems, rendering artifacts, clipping, or other clearly abnormal behavior).

Your job is:

1. Decide whether the clip contains ANY visible bug or glitch based only on what you can actually see in the video.
2. If you think there IS a clear glitch, briefly describe what is wrong in one short sentence.
3. If you think there is NO clear glitch, briefly explain in one short sentence that the behavior looks normal or plausible for this kind of game.

If the behavior is ambiguous and could reasonably be normal gameplay, treat it as **no glitch**.

Then respond strictly in the following JSON format:

{{
  "reasoning": "very short explanation of why you think there is or is not a glitch",
  "glitch_detected": true or false,
  "description_of_glitch": "very short description of the glitch or abnormal behavior; empty string if no glitch"
}}

Do NOT include any extra text outside the JSON.
"""


WITH_PROPS_PROMPT_TEMPLATE = """You are a helpful assistant analyzing short video clips from the video game "{game}".

Below is a checklist of known glitch patterns for this game. Use it to systematically scan the video. For each item, briefly consider whether the video shows that specific violation.

GLITCH CHECKLIST:
{checklist}

INSTRUCTIONS:
1. Scan the video carefully, keeping the checklist in mind.
2. If you see something that matches ANY checklist item, or any other clearly abnormal visual behavior, report it as a glitch.
3. If nothing in the video matches any checklist item and everything looks like plausible normal gameplay, report no glitch.
4. Do NOT hallucinate glitches. Only report what you can actually see. Ambiguous or borderline cases should be treated as no glitch.

Respond strictly in the following JSON format:

{{
  "reasoning": "which checklist items you considered and why you think there is or is not a glitch; cite checklist IDs here if applicable",
  "glitch_detected": true or false,
  "description_of_glitch": "describe IN YOUR OWN WORDS what you actually see going wrong in the video (e.g. 'a car clips through a concrete barrier' or 'the NPC floats into the sky'). Do NOT just write checklist IDs or category names. Empty string if no glitch."
}}

Do NOT include any extra text outside the JSON.
"""


def build_prompt_baseline(game: str) -> str:
    """Baseline prompt (no checklist)."""
    return BASE_PROMPT_TEMPLATE.format(game=game)


def build_prompt_generic(game: str) -> str:
    """GlitchGuard_g prompt: the generic checklist."""
    checklist = _checklist_from_path(os.path.join(PROPERTIES_DIR, GENERIC_FILE), checklist_only=False)
    if not checklist:
        print(f"[WARN] No generic checklist available, using the baseline prompt for '{game}'.")
        return BASE_PROMPT_TEMPLATE.format(game=game)
    return WITH_PROPS_PROMPT_TEMPLATE.format(game=game, checklist=checklist)


def build_prompt_specific(game: str) -> str:
    """GlitchGuard_s prompt: the game-specific checklist of the clip's title."""
    fname = GAME_TO_SPECIFIC_FILE.get(game)
    if not fname:
        print(f"[WARN] No game-specific checklist for '{game}', using the baseline prompt.")
        return BASE_PROMPT_TEMPLATE.format(game=game)
    checklist = _checklist_from_path(os.path.join(PROPERTIES_DIR, fname), checklist_only=False)
    if not checklist:
        print(f"[WARN] Empty or missing game-specific checklist for '{game}', using the baseline prompt.")
        return BASE_PROMPT_TEMPLATE.format(game=game)
    return WITH_PROPS_PROMPT_TEMPLATE.format(game=game, checklist=checklist)


BUILDERS = {
    "baseline": build_prompt_baseline,
    "generic": build_prompt_generic,
    "specific": build_prompt_specific,
}
