"""Building splits (CONTRACTS section 3.1)."""
from typing import List, Optional, Sequence, Tuple

from drive2sim.core.schema import SCHEMA_VERSIONS, Split, parse_strategy


def every_nth_frame_pairs(frame_range: Tuple[int, int], cameras: Sequence[str],
                          n: int) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Train and test pairs for the every_nth_frame strategy.

    Frames first+N, first+2N, ... are test for every camera; all other frames, including the first,
    are train. This matches split_train_test with test_image_stride in drive2sim/recon/data/driving_dataset.py.
    """
    first, last = frame_range
    if n < 1:
        raise ValueError("n must be >= 1")
    test_frames = set(range(first + n, last + 1, n))
    train = [(f, c) for f in range(first, last + 1) if f not in test_frames for c in cameras]
    test = [(f, c) for f in sorted(test_frames) for c in cameras]
    return train, test


def make_split(scene_id: str, cameras: Sequence[str], frame_range: Tuple[int, int],
               strategy: str, split_id: Optional[str] = None) -> Split:
    """Create a split for the every_nth_frame strategy. Custom splits are written by hand."""
    name, n = parse_strategy(strategy)
    if name != "every_nth_frame":
        raise ValueError("make_split builds every_nth_frame splits only; write custom splits directly")
    train, test = every_nth_frame_pairs(frame_range, cameras, n)
    return Split(
        schema_version=SCHEMA_VERSIONS["split"],
        split_id=split_id or f"every{n}_cams{len(cameras)}",
        scene_id=scene_id,
        strategy=strategy,
        frame_range=frame_range,
        cameras=list(cameras),
        train=train,
        test=test,
    )


def matches_strategy(split: Split) -> bool:
    """True if an every_nth_frame split lists exactly the pairs its strategy produces."""
    name, n = parse_strategy(split.strategy)
    if name != "every_nth_frame":
        return True
    train, test = every_nth_frame_pairs(split.frame_range, split.cameras, n)
    return set(train) == set(split.train) and set(test) == set(split.test)
