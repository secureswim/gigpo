# Weighted GiGPO checkpoint reevaluation

All three checkpoints were trained with seed 0 and evaluated at step 150 on ALFWorld valid_seen.
The weighted evaluation on Omni allocation 11018 reused the five decoding seeds and checked the 128 game IDs per seed, in slot order, against both original evaluations.
Some game IDs recur within and across seeds. These are evaluation seeds for fixed checkpoints.

| Evaluation seed | Baseline | Threshold | Weighted |
| --- | ---: | ---: | ---: |
| 1 | 109/128 | 116/128 | 99/128 |
| 2 | 115/128 | 108/128 | 95/128 |
| 3 | 111/128 | 111/128 | 101/128 |
| 4 | 107/128 | 101/128 | 97/128 |
| 5 | 115/128 | 104/128 | 106/128 |
| Total | 557/640 (87.0%) | 540/640 (84.4%) | 498/640 (77.8%) |

The weighted model was trained with tau 1.17512. Each seed directory includes the validation log, episode outcomes, and verification results.

[Original paired evaluation](../paired-validation-10605/)
