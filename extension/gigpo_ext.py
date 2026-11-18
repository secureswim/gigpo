"""Anchor-group diagnostics and selection for GiGPO (course-project extension).

This module is copied into the patched training tree as ``gigpo/gigpo_ext.py``
by ``setup/build_ext_tree.sh``. The pristine snapshot in ``upstream-verl-agent/``
is never edited.

What it adds on top of ``gigpo.core_gigpo.compute_gigpo_outcome_advantage``:

* Diagnostics of every step-level (anchor) group formed in a training batch:
  group sizes, spread of discounted returns, share of groups that carry no
  centred signal, and the magnitude of micro vs. macro advantages.
* Three ways of using the groups once they are formed (``group_filter``):

  ``none``       original GiGPO. The advantage is computed with the original
                 upstream functions and is bit-identical to upstream.
  ``threshold``  condition (b). The micro (step) advantage of a group is set to
                 zero when the group has fewer than ``min_group_size`` members
                 or its return standard deviation is below ``min_return_std``.
  ``weighted``   condition (c). The micro advantage of each group is scaled by
                     w_g = s_g / (s_g + tau) * (n_g - 1) / n_g
                 where s_g is the group's return std, n_g its size and
                 tau = ``weight_tau``. Weights are then rescaled by one batch-wide
                 constant so that the total micro-advantage mass
                 sum_i |A_step_i| is unchanged (the update magnitude is not
                 silently changed; only its distribution across groups).

In every condition the episode (macro) advantage is left untouched.

Why exactly-degenerate groups are not the target: the step advantage is centred
on the group mean, so a group of size one, or a group whose returns are all
equal, already receives zero micro advantage in the original code. Filtering
those is a no-op. What can matter are small or near-zero-spread groups: with
``mean_std_norm`` the normalised advantage is scale-free, so a group of size 2
always gets +-1/sqrt(2) whatever its return gap is, and a tiny return gap gets
the same push as a large one. Note that a small gap is not automatically noise:
with gamma=0.95 two successful trajectories that differ by one step have
returns ~5% apart, which is real efficiency signal. Whether removing such
groups helps is the empirical question of the project.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from gigpo import core_gigpo

VALID_FILTERS = ("none", "threshold", "weighted")
ZERO_STD = 1e-6  # spread below this is treated as "all returns equal"
STD_REFERENCE_LEVELS = (0.1, 0.5, 1.0)  # fixed levels reported for every run


# ---------------------------------------------------------------- #
# Group statistics
# ---------------------------------------------------------------- #
def group_statistics(values: np.ndarray, group_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-group size, std (unbiased, as torch.std in core_gigpo), range, mean, min, max.

    Returns arrays aligned to ``unique_ids`` and, for each sample, the position
    of its group in those arrays (``sample_group``).
    """
    values = np.asarray(values, dtype=np.float64)
    members = defaultdict(list)
    for i, gid in enumerate(group_ids):
        members[gid].append(i)
    unique_ids = list(members.keys())
    sizes = np.array([len(members[g]) for g in unique_ids], dtype=np.int64)
    stds = np.zeros(len(unique_ids))
    ranges = np.zeros(len(unique_ids))
    means = np.zeros(len(unique_ids))
    mins = np.zeros(len(unique_ids))
    maxs = np.zeros(len(unique_ids))
    sample_group = np.empty(len(values), dtype=np.int64)
    for k, gid in enumerate(unique_ids):
        idx = members[gid]
        v = values[idx]
        means[k] = v.mean()
        mins[k] = v.min()
        maxs[k] = v.max()
        ranges[k] = maxs[k] - mins[k]
        stds[k] = v.std(ddof=1) if len(v) > 1 else 0.0
        sample_group[idx] = k
    return {"unique_ids": np.array(unique_ids, dtype=object), "sizes": sizes, "stds": stds,
            "ranges": ranges, "means": means, "mins": mins, "maxs": maxs, "sample_group": sample_group}


def group_weights(stats: Dict[str, np.ndarray], group_filter: str, min_group_size: int = 2,
                  min_return_std: float = 0.0, weight_tau: Optional[float] = None) -> np.ndarray:
    """Un-normalised per-group multiplier on the micro advantage."""
    sizes, stds = stats["sizes"], stats["stds"]
    if group_filter == "none":
        return np.ones(len(sizes))
    if group_filter == "threshold":
        keep = (sizes >= max(int(min_group_size), 2)) & (stds >= float(min_return_std))
        return keep.astype(np.float64)
    if group_filter == "weighted":
        if weight_tau is None or float(weight_tau) <= 0:
            raise ValueError("group_filter=weighted needs algorithm.gigpo.weight_tau > 0 "
                             "(set it with analysis/choose_thresholds.py)")
        tau = float(weight_tau)
        w = stds / (stds + tau) * (sizes - 1) / np.maximum(sizes, 1)
        w[sizes < 2] = 0.0
        return w
    raise ValueError(f"Unknown group_filter {group_filter!r}; expected one of {VALID_FILTERS}")


def _per_sample_scalar(adv_tiled: torch.Tensor, eos_mask: torch.Tensor) -> np.ndarray:
    """Recover the per-sample scalar from a (bs, L) tiled advantage."""
    mask = eos_mask.to(adv_tiled.dtype)
    denom = mask.sum(-1).clamp(min=1)
    return ((adv_tiled * mask).sum(-1) / denom).detach().cpu().double().numpy()


# ---------------------------------------------------------------- #
# Main entry point used by the patched trainer
# ---------------------------------------------------------------- #
def compute_gigpo_outcome_advantage_ext(token_level_rewards: torch.Tensor,
                                        step_rewards: torch.Tensor,
                                        eos_mask: torch.Tensor,
                                        anchor_obs: np.ndarray,
                                        index: np.ndarray,
                                        epsilon: float = 1e-6,
                                        step_advantage_w: float = 1.0,
                                        mode: str = "mean_norm",
                                        group_filter: str = "none",
                                        min_group_size: int = 2,
                                        min_return_std: float = 0.0,
                                        weight_tau: Optional[float] = None,
                                        diagnostics: bool = True,
                                        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, list]]:
    """GiGPO advantage with optional anchor-group selection.

    Returns ``(advantages, returns, metrics, record)``. ``record`` holds the
    per-group arrays for offline analysis (written by :func:`dump_record`).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if group_filter not in VALID_FILTERS:
        raise ValueError(f"Unknown group_filter {group_filter!r}; expected one of {VALID_FILTERS}")

    # Original upstream computations, unchanged.
    episode_adv = core_gigpo.episode_norm_reward(token_level_rewards, eos_mask, index, epsilon, remove_std)
    step_group_uids = core_gigpo.build_step_group(anchor_obs, index)
    step_adv = core_gigpo.step_norm_reward(step_rewards, eos_mask, step_group_uids, epsilon, remove_std)

    returns_np = step_rewards.detach().cpu().double().numpy()
    stats = group_statistics(returns_np, step_group_uids)
    raw_w = group_weights(stats, group_filter, min_group_size, min_return_std, weight_tau)

    step_scalar = _per_sample_scalar(step_adv, eos_mask)
    sample_w = raw_w[stats["sample_group"]]
    mass_before = np.abs(step_scalar).sum()
    scale = 1.0
    if group_filter == "weighted":
        mass_after = np.abs(step_scalar * sample_w).sum()
        scale = mass_before / mass_after if mass_after > 0 else 0.0
        sample_w = sample_w * scale

    if group_filter == "none":
        scores = episode_adv + step_advantage_w * step_adv  # identical to upstream
    else:
        w_t = torch.as_tensor(sample_w, dtype=step_adv.dtype, device=step_adv.device).unsqueeze(-1)
        scores = episode_adv + step_advantage_w * step_adv * w_t

    metrics: Dict[str, float] = {}
    record: Dict[str, list] = {}
    if diagnostics:
        metrics, record = _diagnostics(stats, raw_w, sample_w, step_scalar,
                                       _per_sample_scalar(episode_adv, eos_mask),
                                       token_level_rewards.sum(-1).detach().cpu().double().numpy(),
                                       index, scale, group_filter)
    return scores, scores, metrics, record


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _diagnostics(stats, raw_w, sample_w, step_scalar, episode_scalar, episode_scores, index,
                 scale, group_filter):
    sizes, stds, ranges = stats["sizes"], stats["stds"], stats["ranges"]
    n_groups = len(sizes)
    n_samples = int(sizes.sum())
    multi = sizes >= 2
    zero_signal = (~multi) | (stds <= ZERO_STD)
    spread = multi & (stds > ZERO_STD)
    sample_sizes = sizes[stats["sample_group"]]
    sample_zero = zero_signal[stats["sample_group"]]

    # Episode-level groups: the unit DAPO-style dynamic sampling filters.
    ep_groups = defaultdict(list)
    for i, gid in enumerate(index):
        ep_groups[gid].append(episode_scores[i])
    ep_degenerate = np.mean([np.ptp(v) <= ZERO_STD for v in ep_groups.values()]) if ep_groups else float("nan")

    m = {
        "gigpo_diag/num_step_groups": float(n_groups),
        "gigpo_diag/num_step_samples": float(n_samples),
        "gigpo_diag/mean_group_size": float(sizes.mean()) if n_groups else float("nan"),
        "gigpo_diag/max_group_size": float(sizes.max()) if n_groups else float("nan"),
        "gigpo_diag/frac_groups_size1": float(np.mean(sizes == 1)) if n_groups else float("nan"),
        "gigpo_diag/frac_groups_size2": float(np.mean(sizes == 2)) if n_groups else float("nan"),
        "gigpo_diag/frac_groups_size_ge3": float(np.mean(sizes >= 3)) if n_groups else float("nan"),
        "gigpo_diag/frac_samples_in_size1": float(np.mean(sample_sizes == 1)) if n_samples else float("nan"),
        "gigpo_diag/frac_groups_zero_std": float(np.mean(multi & (stds <= ZERO_STD))) if n_groups else float("nan"),
        "gigpo_diag/frac_groups_zero_signal": float(np.mean(zero_signal)) if n_groups else float("nan"),
        "gigpo_diag/frac_samples_zero_signal": float(np.mean(sample_zero)) if n_samples else float("nan"),
        "gigpo_diag/return_std_p10": _pct(stds[spread], 10),
        "gigpo_diag/return_std_p25": _pct(stds[spread], 25),
        "gigpo_diag/return_std_p50": _pct(stds[spread], 50),
        "gigpo_diag/return_std_p75": _pct(stds[spread], 75),
        "gigpo_diag/return_std_p90": _pct(stds[spread], 90),
        "gigpo_diag/return_range_p50": _pct(ranges[spread], 50),
        "gigpo_diag/mean_abs_step_adv": float(np.mean(np.abs(step_scalar))),
        "gigpo_diag/mean_abs_episode_adv": float(np.mean(np.abs(episode_scalar))),
        "gigpo_diag/frac_episode_groups_all_equal": float(ep_degenerate),
        # Selection effect (all 1.0 / 0.0 for group_filter=none).
        "gigpo_diag/frac_samples_micro_kept": float(np.mean(sample_w > 0)),
        "gigpo_diag/mean_sample_weight": float(np.mean(sample_w)),
        "gigpo_diag/micro_mass_ratio": float(np.abs(step_scalar * sample_w).sum() / max(np.abs(step_scalar).sum(), 1e-12)),
        "gigpo_diag/weight_scale": float(scale),
    }
    for lv in STD_REFERENCE_LEVELS:
        m[f"gigpo_diag/frac_spread_groups_std_lt_{lv}"] = float(np.mean(stds[spread] < lv)) if spread.any() else float("nan")
    m["gigpo_diag/mean_abs_step_over_episode"] = (
        m["gigpo_diag/mean_abs_step_adv"] / m["gigpo_diag/mean_abs_episode_adv"]
        if m["gigpo_diag/mean_abs_episode_adv"] > 0 else float("nan"))

    record = {
        "filter": group_filter,
        "sizes": sizes.tolist(),
        "stds": np.round(stds, 6).tolist(),
        "ranges": np.round(ranges, 6).tolist(),
        "means": np.round(stats["means"], 6).tolist(),
        # min/max let the offline diagnostic tell success-vs-failure groups apart
        # from groups whose returns differ only slightly (diagnostic only).
        "mins": np.round(stats["mins"], 6).tolist(),
        "maxs": np.round(stats["maxs"], 6).tolist(),
        "weights": np.round(raw_w * (scale if group_filter == "weighted" else 1.0), 6).tolist(),
    }
    return m, record


def dump_record(record: Dict[str, list], out_dir: Optional[str], global_step: int) -> None:
    """Append one JSON line per training step with all group-level arrays."""
    if not record or not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    line = dict(step=int(global_step), **record)
    with open(os.path.join(out_dir, "step_groups.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
