# Anchor-group diagnostic

Source: `../runs/full-10605/gigpo_seed0/gigpo_diag/step_groups.jsonl`, training steps 1..20 (20 steps found). Rule v2.

## Group overview

- Anchor groups: 13759; step samples: 120576
- Groups of size 1: 34.7%; size 2: 15.7%; size >= 3: 49.7%
- Groups with >= 2 members but equal returns: 51.2%
- Step samples with zero micro advantage already (size 1 or equal returns): 63.7%
- Spread groups: 1942 holding 43807 step samples
- Spread-group return std: p10 0.0352, p25 0.05774, median 1.175, p75 2.467, p90 3.457

## Thresholds (predeclared rule v2)

- MIN_GROUP_SIZE = 2 (no size filter beyond size-1 groups)
- MIN_RETURN_STD = 0.075 (fixed: just above 0.0707 = 0.1/sqrt(2), the largest std of a group whose returns span <= one invalid-action penalty)
- WEIGHT_TAU = 1.17512 (median of spread-group std, steps 1..20)

## Return bands of spread groups

| Band | Groups | Step samples | Median std | Mean weighted-GiGPO raw weight |
| --- | --- | --- | --- | --- |
| Mixed outcome (some later win, some do not) | 60.0% | 58.0% | 2.033 | 0.505 |
| All later win, returns differ by timing (std >= MIN_RETURN_STD) | 9.2% | 1.7% | 1.238 | 0.321 |
| Penalty-scale spread (std < MIN_RETURN_STD) -- removed by gigpo_threshold | 30.8% | 40.3% | 0.04144 | 0.026 |

Percentages are of spread groups and of their step samples. The raw weight is s/(s+tau)*(n-1)/n before the batch-wide rescaling that keeps total micro mass fixed.

Of the penalty-scale groups, 0.5% are success-only (timing differences far from the goal or one invalid action); the rest never win and differ only by the invalid-action penalty.

### Band shares by training-step block

| Steps | mixed | timing | penalty |
| --- | --- | --- | --- |
| 1-5 | 35.7% | 6.2% | 58.1% |
| 6-10 | 72.9% | 9.4% | 17.6% |
| 11-15 | 82.4% | 12.2% | 5.4% |
| 16-20 | 80.8% | 12.8% | 6.5% |

Shares of spread groups. A drift across blocks means the steps 1..N mix may not represent the rest of training; report it, do not re-derive thresholds.

## Threshold impact forecast

The threshold filter would remove the micro advantage of 40.3% of the step samples that currently receive one (low-impact marker: 5%).

**Noticeable threshold impact expected**: train both extensions.

For comparison only, retired rule v1 (MIN_GROUP_SIZE=3, MIN_RETURN_STD=p25=0.05774) would have removed 40.6% of these step samples.

Group-size counts: 1: 4768, 2: 2154, 3: 1252, 4: 866, 5: 635, 6: 499, 7: 424, 8: 607, 9: 259, 10: 238, 11: 188, 12: 149, 13: 138, 14: 119, 15: 101, 16: 87, 17: 68, 18: 68, 19: 59, 20: 63, 21: 46, 22: 51, 23: 52, 24: 54, 25: 33, 26: 30, 27: 38, 28: 22, 29: 22, 30: 23, 31: 21, 32: 22, 33: 16, 34: 12, 35: 19, 36: 12, 37: 18, 38: 14, 39: 13, 40: 6, 41: 8, 42: 13, 43: 14, 44: 6, 45: 9, 46: 8, 47: 10, 48: 4, 49: 5, 50: 5, 51: 5, 52: 5, 53: 4, 54: 5, 55: 4, 56: 6, 57: 4, 58: 5, 59: 3, 60: 2, 61: 2, 62: 1, 63: 2, 64: 2, 65: 2, 66: 6, 67: 1, 68: 1, 70: 2, 71: 3, 72: 5, 74: 1, 75: 1, 77: 3, 78: 5, 79: 2, 80: 1, 81: 2, 83: 2, 84: 1, 85: 1, 86: 1, 87: 1, 88: 1, 89: 1, 90: 3, 92: 2, 93: 1, 94: 2, 95: 2, 96: 1, 97: 1, 98: 1, 99: 2, 100: 1, 101: 1, 102: 3, 103: 1, 104: 4, 105: 1, 106: 8, 107: 1, 108: 2, 109: 3, 110: 2, 112: 4, 113: 2, 114: 1, 115: 2, 117: 4, 118: 2, 119: 3, 120: 5, 121: 5, 122: 6, 123: 1, 124: 2, 125: 6, 126: 1, 127: 6, 128: 7, 129: 4, 130: 1, 131: 2, 132: 6, 133: 1, 134: 5, 135: 5, 136: 6, 137: 4, 138: 4, 139: 5, 140: 5, 141: 2, 142: 8, 143: 4, 144: 2, 145: 4, 146: 6, 147: 4, 148: 4, 149: 3, 150: 3, 151: 1, 152: 4, 153: 7, 154: 5, 155: 4, 156: 3, 157: 5, 158: 6, 159: 5, 160: 4, 161: 3, 162: 4, 163: 2, 164: 3, 165: 1, 166: 2, 167: 2, 168: 3, 169: 2, 170: 1, 172: 1, 173: 2, 174: 2, 175: 1, 176: 1, 177: 4, 178: 3, 179: 1, 180: 4, 181: 3, 182: 2, 183: 2, 184: 2, 185: 1, 187: 1, 188: 5, 191: 1, 192: 4, 193: 2, 194: 2, 195: 1, 196: 1, 197: 1, 199: 1, 200: 1, 201: 2, 202: 2, 207: 1, 208: 1, 209: 1, 210: 2, 211: 1, 215: 1, 223: 1, 237: 1, 240: 1, 260: 1
