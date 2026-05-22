# Dataset Train/Val/Test Splits

All splits are **bag-disjoint**: every image from a given bag ends up in exactly one split,
preventing temporal leakage between training and evaluation data.

The split assignment for each dataset was determined by an exhaustive search over all
possible bag combinations, optimising for balanced image counts, positive-image counts,
background-image counts, and total annotation instances across train/val/test.

---

## Vision — `solaqua_fish`

**Notebook:** `data-processing/vision/solaqua_fish/solaqua_fish.ipynb`  
**Bag metadata:** `data-processing/vision/SOLAQUA/solaqua_labeled_fish.ipynb`  
**Classes:** `fish`  
**Split ratios (target):** 65 / 15 / 20

| Split | Bags | Images | Instances |
|-------|------|--------|-----------|
| train | 7    | 505    | 2 101     |
| val   | 2    | 124    | 465       |
| test  | 3    | 164    | 587       |
| **total** | **12** | **793** | **3 153** |

| Split | Bag ID | Sequence |
|-------|--------|----------|
| train | bag1   | 2024-08-20_13-55-34 |
| train | bag4   | 2024-08-20_14-31-29 |
| train | bag8   | 2024-08-20_15-20-29 |
| train | bag9   | 2024-08-20_16-34-34 |
| train | bag12  | 2024-08-20_16-47-54 |
| train | bag13  | 2024-08-20_17-02-00 |
| train | bag14  | 2024-08-20_17-14-36 |
| val   | bag2   | 2024-08-20_13-57-42 |
| val   | bag10  | 2024-08-20_16-43-25 |
| test  | bag3   | 2024-08-20_14-16-05 |
| test  | bag11  | 2024-08-20_16-45-21 |
| test  | bag15  | 2024-08-20_17-55-40 |

---

## Sonar — `net_fish_sonar_improved`

**Notebook:** `data-processing/sonar/net_fish_sonar_improved/net_fish_sonar_improved.ipynb`  
**Bag metadata:** `data-processing/sonar/net_fish_sonar/net_fish_sonar.ipynb`  
**Classes:** `fish`, `net`  
**Split ratios (target):** 60 / 20 / 20

This dataset is an improved version of `net_fish_sonar`. The original `bag10`
(200 manually selected images) is replaced by 336 frames derived directly from the
MOT ground-truth annotations for the same sequence (`2024-08-20_17-14-36`), selecting
every frame that contains at least one fish annotation. All other bags (bag1–bag9) are
carried over unchanged from `net_fish_sonar`.

| Split | Bags | Images | Instances |
|-------|------|--------|-----------|
| train | 6    | 1 336  | 1 973     |
| val   | 2    | 400    | 541       |
| test  | 2    | 400    | 565       |
| **total** | **10** | **2 136** | **3 079** |

| Split | Bag ID | Sequence |
|-------|--------|----------|
| train | bag3   | 2024-08-20_13-55-34 |
| train | bag4   | 2024-08-20_14-31-29 |
| train | bag6   | 2024-08-20_14-34-07 |
| train | bag7   | 2024-08-20_14-16-05 |
| train | bag8   | 2024-08-20_18-50-22 |
| train | bag10  | 2024-08-20_17-14-36 |
| val   | bag2   | 2024-08-20_17-02-00 |
| val   | bag5   | 2024-08-20_14-38-37 |
| test  | bag1   | 2024-08-20_13-57-42 |
| test  | bag9   | 2024-08-20_16-47-54 |


---

**TODO:** 
Sequences that are present here in training should not be present in the final MOT sequences. IT IS NOW, and these MOT sequences are therefor removed from the MOT sequences. Therefor there is a TODO to annotate new MOT sequences that are not in training splits for detection. 

Sequences that are in train for detection and also in MOT:

2024-08-20_14-31-29, for VISION and SONAR => this is not removed from MOT as it has many frames, few frames in detection. But still a slight limitatoin and should be fixed in future work.

2024-08-20_17-02-00, for VISION, val in train => this shuold be REMOVED from MOT

2024-08-20_17-14-36, for VISION and SONAR => many overlapping, this is REMOVED from MOT

2024-08-20_17-39-32, NEITHER => OK
2024-08-20_17-55-40, NEITHER in train => OK
