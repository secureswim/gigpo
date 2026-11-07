# Linux training-machine handoff

After the owner grants access to this private repository, clone it on the Linux machine:

```bash
git clone https://github.com/secureswim/gigpo.git
cd gigpo
```

Run the remaining commands from this project root. The vendored upstream source is already included at the paper-era commit; do not update it before the baseline experiments.

## 1. Verify the source

```bash
cat upstream-verl-agent/UPSTREAM_COMMIT
python3 week1/check_baseline_scripts.py
```

The hash must be `f974dc5977908d6007ad178c9f3c48f0c7b39331`. The checker should report matching shared settings and only method-specific differences.

## 2. Install the pinned version's dependencies

Use a Linux machine with an NVIDIA CUDA setup compatible with the packages below. The historical README specifies:

```bash
cd upstream-verl-agent
conda create -n verl-agent-gigpo python=3.12 -y
conda activate verl-agent-gigpo
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install flash-attn --no-build-isolation
python -m pip install -e .
python -m pip install vllm==0.8.2
python -m pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld thinc==8.3.4
cd ..
```

Follow the repository's pinned README if a dependency needs adjustment for that machine. Record any deviation with its reason. Do not mix the current README's newer vLLM version with this historical code without testing it.

## 3. Download and test ALFWorld

```bash
alfworld-download -f
python3 week1/check_alfworld.py
```

The automated check resets the text environment, takes one admissible action, and prints the observations. Run it from the project root. `alfworld-play-tw` is available if you also want to inspect a game manually. The smoke script uses the text environment; no robot is needed for this check.

## 4. Run one GiGPO epoch

```bash
bash week1/smoke_gigpo.sh
```

The script keeps the repository's model and ALFWorld settings, overrides the run to one epoch, enables validation at that step, and uses console logging so a W&B login is unnecessary. It writes `runs/week1_smoke/metadata.txt` and `runs/week1_smoke/train.log`.

Success means the log contains an optimizer update and a validation result. An ALFWorld launch alone is insufficient. The run may download the model and a small placeholder dataset on first use. If it fails, preserve the complete log and record the fix before rerunning.

## 5. Hand back the evidence

Share the two files in `runs/week1_smoke/`, the output of `check_alfworld.py`, and any installation deviations. That lets us diagnose failures and prepare the matched full runs without guessing what happened.
