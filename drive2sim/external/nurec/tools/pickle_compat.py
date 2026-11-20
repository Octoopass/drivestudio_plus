"""pickle.load that also reads pickles written by NumPy 2 when NumPy 1.x is installed.

NumPy 2 renamed its internal package numpy.core to numpy._core, and pickles of NumPy arrays/scalars record that
module path. The KITTI-360 ROSA metadata (kitti360_rosa_infos_*.pkl) may be produced with NumPy 2, while the
DriveStudio+ environment is pinned to NumPy 1.23. Mapping the module name back is enough: the array layout
(numpy.core.multiarray._reconstruct, dtype, scalar) is unchanged between the two.
"""
from __future__ import annotations

import pickle
from typing import BinaryIO


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            import numpy
            if not hasattr(numpy, "_core") or numpy.__version__.startswith("1."):
                module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load(f: BinaryIO):
    return _Unpickler(f).load()
