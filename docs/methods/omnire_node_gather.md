# Faster gathering of OmniRe node Gaussians

## Where the code is

- `drive2sim/recon/models/gaussians/vanilla.py`: `get_gaussians` of the background node (`VanillaGaussians`).
- `drive2sim/recon/models/nodes/rigid.py`: `get_gaussians`, `transform_means` and `transform_quats` of the vehicle node (`RigidNodes`).
- `drive2sim/recon/models/nodes/deformable.py`: the embedding lookup of the deformable node (`DeformableNodes`).
- Test: `tests/drive2sim/recon/models/test_node_gather.py` (CPU, against the original indexing).

## What it does

It removes two expensive indexing operations from every step without changing results.

1. Background, vehicle and deformable nodes return their five per-Gaussian tensors (positions, opacities, colours, scales, rotations) directly, not through an always-true mask. `self.filter_mask` is still stored, so densification's gradient bookkeeping is unchanged.
2. The vehicle node looks up each Gaussian's vehicle pose with `torch.index_select` instead of `tensor[point_ids]`. The deformable node does the same for its embedding.

## Why this method

At step 20 000, OmniRe's own PyTorch work took 77 % of GPU time, and its largest kernel was this indexing backward. Alternatives were:

- A custom CUDA kernel could gather and transform node Gaussians in one pass, with a hand-written backward. It would need compiling on Kaggle, gradient tests and maintenance, and was not written.
- `torch.compile` traces PyTorch code and fuses operations. By default, it breaks the graph at operators with data-dependent output size, such as boolean-mask indexing (`torch._dynamo.config.capture_dynamic_output_shape_ops`). Densification also changes the Gaussian count, which can trigger recompilation. It was not tried.

The chosen change is a few lines of standard PyTorch per node, with no build step.

## First-principles explanation

Indexing in PyTorch (`x[mask]`, `x[idx]`) copies data in the forward pass. The backward scatters the gradient back and sums repeated positions. In an OmniRe profile, this backward kernel (`indexing_backward_kernel_small_stride`) took about 32 ms per step at step 2000 and 98 ms at step 20 000, growing with the Gaussian count.

The first cause was the always-true mask: selecting all elements returns the same tensor, so copy and scatter are pure overhead.

The second cause was the vehicle poses, which hundreds of thousands of Gaussians read from a few dozen vehicles. The backward of `x[idx]` uses `index_put_` with accumulation, which sorts the indices and sums each group of duplicates. This is slow when many indices point to few rows. The backward of `torch.index_select`, `index_add_`, adds atomically and is much faster here.

On a Quadro T2000, forward plus backward for five tensors of one million Gaussians through the mask went from 36.3 ms to 5.8 ms, and the pose lookup of 40 vehicles for 400 000 Gaussians from 8.6 ms to 2.3 ms.

## Difference from the original code, method or paper

- The original DriveStudio (`ziyc/drivestudio`) creates `filter_mask = torch.ones_like(...)` in `get_gaussians` and indexes through it. Comments in `rigid.py` and `smpl.py` ("hack here, need to consider a gaussian filter for efficient rendering") suggest a planned but unwritten filter. The PVG node uses a real mask and is unchanged.
- `torch.index_select` is mathematically identical to the original `pose[point_ids]`.
- On real OmniRe nodes, 52 of 57 output and gradient tensors were bit-identical. The other 5 (vehicle pose gradients and embeddings) differed by at most 2.5e-7 relative, the float32 rounding level. Runs of the original differ this much too, because gsplat also sums gradients atomically.

## Limits, risks and possible gains

- The human node (`SMPLNodes`) has both patterns and is unchanged. Changing it needs the same comparison.
- If a real visibility filter is added, indexing through the mask must return.
- Training is not bit-reproducible, before or after this change. Small differences change which Gaussians cross a densification threshold, so runs give different Gaussian sets with similar metrics.
- The whole-step speed-up on a real run is not measured separately.
