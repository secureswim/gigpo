#!/usr/bin/env bash
# Build the patched training tree used for every run.
#
#   bash setup/build_ext_tree.sh            # -> work/verl-agent-ext
#   EXT_TREE=/scratch/ext bash setup/build_ext_tree.sh
#
# Copies the untouched snapshot in upstream-verl-agent/, adds
# extension/gigpo_ext.py as gigpo/gigpo_ext.py and applies
# extension/verl-agent-ext.patch (trainer hook, env-step counter, config keys).
# With algorithm.gigpo.group_filter=none the advantages equal upstream's
# (tests/test_gigpo_ext.py checks this), so the GiGPO control and GRPO also run
# from this tree and every condition shares one code base.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$PROJECT_ROOT/upstream-verl-agent"
DEST="${EXT_TREE:-$PROJECT_ROOT/work/verl-agent-ext}"
PATCH="$PROJECT_ROOT/extension/verl-agent-ext.patch"
EXT="$PROJECT_ROOT/extension/gigpo_ext.py"

python3 "$PROJECT_ROOT/week1/check_baseline_scripts.py"

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -a "$SRC" "$DEST"
cp "$EXT" "$DEST/gigpo/gigpo_ext.py"

if command -v patch >/dev/null 2>&1; then
  (cd "$DEST" && patch -p1 --forward --quiet < "$PATCH")
else
  (cd "$DEST" && git apply -p1 "$PATCH")
fi

python3 -m py_compile \
  "$DEST/gigpo/gigpo_ext.py" \
  "$DEST/verl/trainer/ppo/ray_trainer.py" \
  "$DEST/agent_system/multi_turn_rollout/rollout_loop.py"

sha() { sha256sum "$1" | cut -d' ' -f1; }
{
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "upstream_commit=$(cat "$SRC/UPSTREAM_COMMIT")"
  echo "patch_sha256=$(sha "$PATCH")"
  echo "gigpo_ext_sha256=$(sha "$EXT")"
} > "$DEST/EXT_BUILD_INFO"

echo "Patched tree ready: $DEST"
cat "$DEST/EXT_BUILD_INFO"
