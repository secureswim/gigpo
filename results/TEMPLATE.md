# Run record: <name>

## Identity

- Date and operator:
- Condition: GiGPO / GRPO / filtered / weighted
- Source commit and any local changes:
- EXT_BUILD_INFO (patch and gigpo_ext hashes, from launches.txt):
- Slurm job IDs (all jobs of the chain) and node(s):
- Credits used:
- Exact command (from launches.txt):
- Log path or URL:
- Checkpoint path or URL:

## Environment

- Partition, GPU and RESOURCE_FLAGS (cluster/omni/resources.env):
- Hardware knobs (N_GPUS, MICRO_BS, GPU_MEM_UTIL, VAL_BATCH, REF_OFFLOAD):
- Python, PyTorch, CUDA, vLLM, ALFWorld versions:
- Installation deviations:
- ALFWorld data version and split:
- Model identifier and revision:
- Random seeds:

## Configuration

- Rollouts per state, batch sizes, episode limit:
- Learning rate, KL setting, discount factor where applicable:
- Training steps or epochs completed:
- Validation frequency and sampling settings:
- Checkpoint policy:
- For extension runs: group_filter, MIN_GROUP_SIZE / MIN_RETURN_STD / WEIGHT_TAU and the diagnostic report they came from:
- Resumed from a checkpoint? At which steps:

## Outcome

- Final validation success rate and exact log metric name:
- Best validation success rate and step:
- Environment steps consumed (sum of env/steps_collected):
- Steps and environment steps to 0.70 success:
- Wall time:
- Episode length, actor/entropy, training reward std, actor/grad_norm:
- Anchor-group diagnostics (gigpo_diag/frac_samples_zero_signal, frac_groups_size2, micro_mass_ratio):
- Repeated evaluation results or other seeds:
- Errors, interruptions, and fixes:

## Interpretation

What does this run establish? What remains uncertain?
