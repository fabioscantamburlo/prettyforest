"""Unit tests for NumPy to Polars data adapter."""

import numpy as np
import pytest

from prettyforest.analysis import numpy_to_polars


class TestNumpyToPolars:
    def test_columns_match_feature_names(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        names = ["a", "b", "c"]
        df = numpy_to_polars(arr, names)

        assert df.columns == names
        assert df.shape == (2, 3)

    def test_values_preserved(self):
        arr = np.array([[1.5, 2.5], [3.5, 4.5]])
        df = numpy_to_polars(arr, ["x", "y"])

        assert df["x"].to_list() == [1.5, 3.5]
        assert df["y"].to_list() == [2.5, 4.5]

    def test_column_count_mismatch_raises(self):
        arr = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="columns"):
            numpy_to_polars(arr, ["a", "b"])

    def test_1d_array_raises(self):
        arr = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="2D"):
            numpy_to_polars(arr, ["a", "b", "c"])
