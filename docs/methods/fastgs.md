# FastGS-style multi-view densification and pruning

An adaptation of the densification and pruning rules of [FastGS](https://github.com/fastgs/FastGS) to DriveStudio.
It changes which Gaussians are added and removed; it does not change the renderer, so it works with
`render_mode: default`, `ut` and `geer`.

Enable it under `trainer` in a method config (see `streetgs_fastgs.yaml`, `streetgs_geer_fastgs.yaml`, `omnire_fastgs.yaml`):

```yaml
trainer:
  fastgs:
    enabled: true
```

## Where the code is

`drive2sim/recon/speedups/fastgs.py` scores the Gaussians (class `FastGS`); `fastgs_rules.py` holds the defaults, the schedule and the view sampling, and is tested on CPU in `tests/drive2sim/recon/speedups/`. The trainer calls `FastGS.before_step` at the start of a step and `FastGS.clear_masks` after the models refine; the models apply the masks in `refinement_after()` (`fastgs_filter_densify`, `fastgs_extra_culls` in `models/gaussians/vanilla.py`).

## What it does

At every refinement step that densifies or prunes, the trainer first renders `num_views` training views
spread evenly over the clip. In each view it marks the high-error pixels (min-max normalised L1 error above
`loss_thresh`) and measures how much each Gaussian contributes to them.

- **Densification.** A Gaussian is split or duplicated only if it passes DriveStudio's usual gradient test
  *and* its score exceeds `score_thresh`.
- **Pruning.** From `prune_from` on, at the last refinement before each opacity reset, Gaussians with opacity
  below `prune_min_opacity` or a normalised error score above `prune_score_thresh` are removed, in addition
  to DriveStudio's own culling.

Applied to `Background`, `RigidNodes` and `DeformableNodes`. `SMPLNodes` and PVG are unchanged.

## Differences from the paper

1. **Blend-weighted count.** FastGS counts each high-error pixel a Gaussian touches as 1, using its own
   rasterizer. Here each pixel counts with the Gaussian's blending weight (alpha x transmittance), obtained
   as a gradient through the existing rasterizer. FastGS's threshold of 5 therefore does not carry over.
2. **Averaged over the views that see the Gaussian.** FastGS averages over all sampled views. In a
   forward-driving clip most views do not see a given Gaussian, so that rule would block nearly all
   densification.
3. **No pruning budget during densification**, and none of FastGS's rasterizer speed-ups.

## Tuning

Every scoring pass logs a line such as

```
FastGS @ step 1200 Background: seen in sampled views 412345/500000, may densify 30112, score p50/p90/p99 0.21/1.80/9.40
```

Set `score_thresh` from these percentiles. If "seen in sampled views" is low, raise `num_views` or set
`densify_unseen: true`. Cost is about two extra renders per sampled view per refinement step.
