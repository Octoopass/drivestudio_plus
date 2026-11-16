# Faster gathering of OmniRe node Gaussians

## Where the code is

- `drive2sim/recon/models/gaussians/vanilla.py`: `get_gaussians` of the background node (`VanillaGaussians`).
- `drive2sim/recon/models/nodes/rigid.py`: `get_gaussians`, `transform_means` and `transform_quats` of the vehicle node (`RigidNodes`).
- `drive2sim/recon/models/nodes/deformable.py`: the per-Gaussian embedding lookup of the deformable node (`DeformableNodes`).
- Test: `tests/drive2sim/recon/models/test_node_gather.py` (CPU, compared with the original indexing).

## What it does

It removes two expensive indexing operations from every training step without changing the results.

1. The background, vehicle and deformable nodes no longer copy their five per-Gaussian tensors (positions, opacities, colours, scales, rotations) through a mask that is always true. They return the tensors directly. `self.filter_mask` is still stored, so the gradient bookkeeping for densification is unchanged.
2. The vehicle node looks up each Gaussian's vehicle pose with `torch.index_select` instead of `tensor[point_ids]`. The deformable node does the same for its per-instance embedding.

## First-principles explanation

Indexing a tensor with a mask or an index list in PyTorch (`x[mask]`, `x[idx]`) copies data in the forward pass. In the backward pass the gradient has to be scattered back to the original positions, and positions that appear several times must be summed. A profile of OmniRe showed that this backward kernel (`indexing_backward_kernel_small_stride`) was the largest single kernel, about 32 ms per step at step 2000 and about 98 ms per step at step 20 000, because its cost grows with the number of Gaussians.

The first cause was a mask that is always true. Selecting all elements gives back the same tensor, so the copy and the scattered gradient are pure overhead.

The second cause was the vehicle poses. Hundreds of thousands of Gaussians read their pose from a few dozen vehicles. The backward of `x[idx]` uses `index_put_` with accumulation, which sorts the indices and then sums each group of duplicates. This is slow when many indices point to few rows. `torch.index_select` computes the same values, and its backward uses `index_add_`, which adds with atomic operations and is much faster for this pattern.

On a Quadro T2000, forward plus backward: five tensors of one million Gaussians through the mask went from 36.3 ms to 5.8 ms; poses of 40 vehicles for 400 000 Gaussians went from 8.6 ms to 2.3 ms.

## Difference from the original code, method or paper

- The original DriveStudio code (`ziyc/drivestudio`) creates `filter_mask = torch.ones_like(...)` in `get_gaussians` and indexes through it. Comments in `rigid.py` and `smpl.py` ("hack here, need to consider a gaussian filter for efficient rendering") suggest that a real filter was planned but not written. The PVG node uses a real mask and is not changed.
- The original code indexes vehicle poses with `pose[point_ids]`. The change uses `torch.index_select`, which is mathematically identical.
- Outputs and gradients were compared on real OmniRe node objects: 52 of 57 tensors were identical bit for bit, and 5 tensors (vehicle pose gradients and embeddings) differed by at most 2.5e-7 relative. This is the rounding level of float32. The original code already has differences of this size between runs, because the gsplat rasterizer also sums gradients with atomic operations.

## Limits, risks and possible gains

- The human node (`SMPLNodes`) has the same two patterns and is not changed yet. Changing it needs the same output and gradient comparison for human nodes.
- If a real visibility filter is added later, indexing through the mask must be restored.
- Training is not bit-reproducible before or after the change. Small differences can change which Gaussians cross a densification threshold, so two runs give different Gaussian sets with similar metrics.
- The speed-up of a whole step on a real run is not measured separately.
