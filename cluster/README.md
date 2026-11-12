# Omni single-MIG smoke run

Preparation runs on the internet-connected master in a new environment at
`/home/vansh23580/rl-project/envs/gigpo`. Existing Conda environments are preserved.
Logs: `/home/vansh23580/rl-project/logs/gigpo-prepare.log`.

The historical vLLM 0.8.2 dependency xgrammar 0.1.16 is unavailable from the
package index. Build the official tag, resolved to
`989222175c2a30fb7987d8bcce35bec1bf6817f2`, instead of substituting a newer version.

`smoke-omni.sh` retains the upstream GiGPO algorithm and task settings, with
one GPU/tensor-parallel rank, microbatches of one, four prompt groups of eight
trajectories (32 total), PPO minibatches of 32 transitions, validation batches of
eight, and a vLLM memory fraction of 0.45. Actor/reference parameters and optimizer
remain on GPU. One training update is followed by validation; initial validation
is skipped. This is a pipeline check, not a paper-result reproduction. The initial
128-trajectory profile with CPU offload exceeded the job's 64 GiB host-RAM limit
while computing reference log probabilities; its logs are preserved separately.

The runtime copy of the upstream shell script skips data preprocessing (already
completed on master), and consumes the engine positional argument before passing
Hydra overrides. The upstream script itself is unchanged.

`check_baseline_scripts.py` accepts the original CRLF manifest or its Git LF
checkout, while continuing to validate file contents and shared settings.

Compatibility deviation: Thinc 8.3.4 pulls NumPy 2, conflicting with vLLM 0.8.2.
Use Thinc 8.2.5, spaCy 3.7.5, NumPy 1.26.4, and contourpy 1.3.2 instead.
TextWorld 1.7.0's locally built wheel has malformed platform-tag metadata that
`pip check` flags; the actual ALFWorld reset/step runtime test passed.
