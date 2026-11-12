# GiGPO single-MIG smoke: PASS

12 November 2026. Slurm job **10333**, successful training step **10333.13**,
`COMPLETED`, exit `0:0`. Successful attempt: 19:39:17–20:12:27 IST
(33 minutes 10 seconds, including initialization and checks).

## Evidence

- Local training output: [train.txt](train.txt), including `step:1`,
  nonzero `actor/grad_norm=4.306`, and final validation metrics.
- `train.txt` removes terminal color codes and trailing whitespace only. Original
  raw-log SHA-256: `309e0f5e575ad4660aa868c8e9adef4d95a201fff853b141bc6019e3f78f7102`.
- Exact executed baseline copy: [baseline-offline.sh](baseline-offline.sh);
  overrides: [smoke-omni.sh](../../cluster/smoke-omni.sh); packages: [packages.txt](packages.txt).
- Remote master log: `/home/vansh23580/rl-project/logs/gigpo-10333.log`.
- Remote exit marker: `/home/vansh23580/rl-project/logs/gigpo-10333.status`.
- CUDA matmul and compute-node ALFWorld reset/step both passed.

## Hardware and configuration

- `short`, `dgxh200`, one H200 MIG `3g.71gb`, 69.75 GiB usable VRAM,
  10 CPUs and 64 GiB host RAM. Cost: 0.10 credits; remaining group balance 34.90.
- Isolated environment: `/home/vansh23580/rl-project/envs/gigpo`.
- PyTorch 2.6.0+cu124, vLLM 0.8.2, Transformers 4.51.1, Ray 2.43.0,
  ALFWorld 0.4.2. Complete versions are in the package manifest.
- Upstream revision: `f974dc5977908d6007ad178c9f3c48f0c7b39331`.
- Qwen/Qwen2.5-1.5B-Instruct revision:
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (3,098,973,447 downloaded bytes).

| Batch / sequence setting | Successful smoke value |
| --- | --- |
| Training prompts per outer iteration | 4 |
| Rollouts per prompt (`env.rollout.n`) | 8 |
| Total training trajectories | 32 |
| PPO minibatch size | 32 transitions |
| Actor training GPU microbatch | 1 |
| Actor log-probability GPU microbatch | 1 |
| Reference log-probability GPU microbatch | 1 |
| Validation examples / batch size | 128 / 8 (16 batches) |
| Maximum environment actions | 50 per trajectory |
| Maximum prompt / response length | 2,048 / 512 tokens |
| Outer training iterations completed | 1 |
| Tensor parallel size / GPU count | 1 / 1 |
| vLLM GPU-memory fraction | 0.45 |

- Full model parameters trained; GPU parameters/optimizer, no CPU offload;
  tensor parallelism one, vLLM memory fraction 0.45.
- One outer training iteration, followed by 128 validation examples in batches
  of eight. Validation sampling temperature 0.4; environment seed 0.
- Baseline learning rate 1e-6, KL coefficient 0.01, gamma 0.95,
  GiGPO step weight 1.0, mode mean_std_norm. Checkpoint saving disabled as in
  the baseline; **no trained model checkpoint was saved**.

## Results and timings

- `val/success_rate`: **0.0625 (6.25%)**.
- `val/test_score/text`: 0.2518277822908205.
- Training trajectory success: 0.0625; mean episode length 48.094.
- Training transition batch after alignment: 1,536; valid action ratio 0.811.
- Generation 144.245 s; actor log probabilities 63.166 s;
  reference probabilities 156.182 s; advantages 0.380 s;
  actor update 276.456 s; validation 1,193.200 s.
- Total timed iteration: 1,833.779 s (30.56 min), of which training work was
  about 10.68 min and validation 19.89 min. Initialization adds about 2.6 min.

## Fixes and limitations

Original xgrammar 0.1.16 was built from its official source because its wheel
was missing from the index. FlashAttention uses its official matching wheel.
NumPy 1.26.4 required compatible Thinc/spaCy, PyArrow, pandas, SciPy and plotting
versions; see [constraints.txt](../../cluster/constraints.txt). The baseline checker accepts Git's
LF checkout of its original CRLF-hashed files. TextWorld's locally built wheel
metadata triggers a pip platform warning, while its runtime check passes.

The first training launch failed on PyArrow/NumPy compatibility. The next
128-trajectory, CPU-offload attempt exceeded host RAM during reference
computation. Both attempts' logs remain on the master as
`logs/gigpo-10333-attempt1.log` and `logs/gigpo-10333-attempt2.log`.
No additional allocation was created. The successful run uses the reduced
profile described above, so it is not a matched full baseline or a paper-result
reproduction. No evaluation before training or multi-seed comparison was run.

Job 10333 was verified still RUNNING after training; its scheduled end is
**13 November 2026, 01:18:51 IST**. Master tmux session: `rl-10333`.

For the full 150-iteration baseline on a full B200, 12–48 hours remains an
unmeasured planning range. This reduced single-MIG run cannot establish B200
throughput; benchmark the actual full-run batch configuration before committing
to a more precise estimate.
