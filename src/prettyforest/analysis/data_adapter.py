"""Convert NumPy arrays to Polars DataFrames for flow computation."""

from __future__ import annotations

import numpy as np
import polars as pl


def numpy_to_polars(data: np.ndarray, feature_names: list[str]) -> pl.DataFrame:
    """Convert a NumPy array to a Polars DataFrame using positional feature names."""
    if data.ndim != 2:
        msg = f"Expected 2D array, got {data.ndim}D"
        raise ValueError(msg)
    if data.shape[1] != len(feature_names):
        msg = f"Array has {data.shape[1]} columns but {len(feature_names)} feature names provided"
        raise ValueError(msg)
    return pl.DataFrame({name: data[:, i] for i, name in enumerate(feature_names)})
