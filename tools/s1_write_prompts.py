"""Writes results/s1_prompts.json. Run (and commit) BEFORE s1_zeroshot.py; the candidates are fixed here, before any result is seen."""
import json
from pathlib import Path

SEV, MIL, NON = "Severe damage", "Mild damage", "Little or no damage"
E1 = {SEV: ["a photo of severe damage"], MIL: ["a photo of mild damage"], NON: ["a photo of no damage"]}
E2 = {
    SEV: ["a photo of collapsed or destroyed buildings after a disaster", "a photo of a destroyed road or bridge",
          "a photo of a house reduced to rubble"],
    MIL: ["a photo of a building with a damaged roof or broken windows", "a photo of fallen trees and minor damage after a storm",
          "a photo of a partially damaged structure that is still standing"],
    NON: ["a photo of an undamaged building", "a photo of a street with no visible damage", "a photo of people or a scene with no damage"],
}
# E3: the E2 phrases with their leading "a photo of " removed, wrapped in each of these templates (class = mean over all phrase x template)
E3_TEMPLATES = ["a social media photo of {}", "a picture taken during a natural disaster showing {}"]
Path("results").mkdir(exist_ok=True)
json.dump({"E1": E1, "E2": E2, "E3": {"templates": E3_TEMPLATES, "phrases_from": "E2 with leading 'a photo of ' removed"}},
          open("results/s1_prompts.json", "w"), indent=2)
print("written")
