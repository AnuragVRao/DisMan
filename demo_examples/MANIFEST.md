# Demo examples

Upload the image and paste the text from the matching `.txt` file into the demo (URLs removed; the model strips them anyway). `_contact_sheet.png` shows all ten at a glance. All images are from the **test split** and have no
tweet/text/image twin in the damage-train index (the neighbour panel cannot show the same tweet). The six built-in *Preloaded examples*
are different images.

**Selection:** picked by true label first, then (for the `good` picks) by the model being right, so these show the system at its best;
the two `FAILURE_*` rows are deliberate failure cases, to show the limits honestly. Overall test accuracy is in `RESULTS.md`, not here.

`n/a` = the image is not in that task's test split, so there is no ground truth for that column.

| # | File | Kind | Event | T1 true -> pred | T2 true -> pred (p) | T3 true -> pred (vote share) | Why show it |
|---|---|---|---|---|---|---|---|
| 1 | `01_severe_damage` | good | california_wildfires | n/a -> informative | Infrastructure and utility damage -> Infrastructure and utility damage (0.99) | Severe damage -> Severe damage (0.8) | Clear severe damage: infrastructure class plus a unanimous, similar-cases vote |
| 2 | `02_mild_damage` | good | hurricane_maria | n/a -> informative | n/a -> Not humanitarian (0.51) | Mild damage -> Mild damage (0.6) | Mild damage is the hardest severity class (18% recall overall); a correct one is worth showing |
| 3 | `03_little_or_no_damage` | good | hurricane_irma | informative -> informative | n/a -> Infrastructure and utility damage (0.72) | Little or no damage -> Little or no damage (0.6) | Little or no damage |
| 4 | `04_rescue_donation` | good | hurricane_harvey | informative -> informative | Rescue, volunteering or donation effort -> Rescue, volunteering or donation effort (0.65) | n/a -> Severe damage (0.8) | Rescue / donation tweet: the Task 2 class that is not about damage |
| 5 | `05_affected_individuals` | good | mexico_earthquake | n/a -> informative | Affected individuals -> Affected individuals (0.95) | n/a -> Severe damage (1.0) | Affected individuals (a small class: 88 test samples) |
| 6 | `06_not_humanitarian` | good | hurricane_maria | not_informative -> not_informative | Not humanitarian -> Not humanitarian (1.00) | n/a -> Mild damage (0.4) | Irrelevant tweet: both Task 1 and Task 2 say so |
| 7 | `07_vehicle_damage` | good | california_wildfires | n/a -> informative | Vehicle damage -> Vehicle damage (0.98) | n/a -> Severe damage (1.0) | Vehicle damage (a small class: 36 test samples) |
| 8 | `08_t1_t2_disagree` | good | hurricane_maria | n/a -> informative | Not humanitarian -> Not humanitarian (0.96) | n/a -> Severe damage (0.8) | Task 1 and Task 2 disagree: shows the disagreement note (text labels vs image labels) |
| 9 | `09_FAILURE_task2` | failure | hurricane_maria | informative -> not_informative | Not humanitarian -> Infrastructure and utility damage (0.75) | n/a -> Severe damage (0.6) | Task 2 confidently wrong: shown on purpose |
| 10 | `10_FAILURE_task3` | failure | hurricane_maria | n/a -> not_informative | n/a -> Infrastructure and utility damage (1.00) | Mild damage -> Severe damage (1.0) | Task 3 unanimous vote but wrong: shows that vote share is not accuracy |
