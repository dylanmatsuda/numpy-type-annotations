"""Tests for ShapeBindingModel."""

import numpy as np
import pytest
from pydantic import ValidationError

from src.ndarray import NDArray
from src.models import ShapeBindingModel


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class VectorPair(ShapeBindingModel):
    a: NDArray[np.float64, ("N",)]
    b: NDArray[np.float64, ("N",)]

class MatMulPair(ShapeBindingModel):
    weights: NDArray[np.float64, ("M", "N")]
    bias: NDArray[np.float64, ("N",)]

class SquareMatrix(ShapeBindingModel):
    matrix: NDArray[np.float64, ("N", "N")]

class TripleN(ShapeBindingModel):
    a: NDArray[np.float32, ("N", 2)]
    b: NDArray[np.float32, ("N", 3)]
    c: NDArray[np.float32, ("N",)]

class IndependentSymbols(ShapeBindingModel):
    x: NDArray[np.float64, ("M",)]
    y: NDArray[np.float64, ("N",)]


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

class TestShapeBindingValid:
    def test_matching_vectors(self):
        m = VectorPair(
            a=np.ones(5, dtype=np.float64),
            b=np.ones(5, dtype=np.float64),
        )
        assert m.a.shape == (5,)
        assert m.b.shape == (5,)

    def test_matmul_pair_consistent_n(self):
        m = MatMulPair(
            weights=np.ones((3, 4), dtype=np.float64),
            bias=np.ones((4,), dtype=np.float64),
        )
        assert m.weights.shape == (3, 4)
        assert m.bias.shape == (4,)

    def test_square_matrix(self):
        m = SquareMatrix(matrix=np.eye(4, dtype=np.float64))
        assert m.matrix.shape == (4, 4)

    def test_triple_n_all_consistent(self):
        m = TripleN(
            a=np.ones((5, 2), dtype=np.float32),
            b=np.ones((5, 3), dtype=np.float32),
            c=np.ones((5,), dtype=np.float32),
        )
        assert m.a.shape[0] == m.b.shape[0] == m.c.shape[0] == 5

    def test_independent_symbols_allow_different_sizes(self):
        m = IndependentSymbols(
            x=np.ones(3, dtype=np.float64),
            y=np.ones(7, dtype=np.float64),
        )
        assert m.x.shape == (3,)
        assert m.y.shape == (7,)

    def test_no_ndarray_fields_passes(self):
        from pydantic import BaseModel as _Base
        class PlainModel(ShapeBindingModel):
            value: int
        assert PlainModel(value=42).value == 42

    def test_single_symbolic_field_any_size(self):
        class Single(ShapeBindingModel):
            arr: NDArray[np.float64, ("N",)]
        for n in [1, 10, 100]:
            m = Single(arr=np.ones(n, dtype=np.float64))
            assert m.arr.shape == (n,)

    def test_size_one_vectors(self):
        m = VectorPair(
            a=np.ones(1, dtype=np.float64),
            b=np.ones(1, dtype=np.float64),
        )
        assert m.a.shape == (1,)
        assert m.b.shape == (1,)


# ---------------------------------------------------------------------------
# Symbol mismatch errors
# ---------------------------------------------------------------------------

class TestShapeBindingErrors:
    def test_vector_pair_mismatch(self):
        with pytest.raises(ValidationError) as exc:
            VectorPair(
                a=np.ones(5, dtype=np.float64),
                b=np.ones(6, dtype=np.float64),
            )
        assert "N" in exc.value.errors()[0]["msg"]

    def test_matmul_bias_mismatch(self):
        with pytest.raises(ValidationError) as exc:
            MatMulPair(
                weights=np.ones((3, 4), dtype=np.float64),
                bias=np.ones((5,), dtype=np.float64),
            )
        msg = exc.value.errors()[0]["msg"]
        assert "N" in msg
        assert "bias" in msg

    def test_non_square_matrix_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SquareMatrix(matrix=np.ones((3, 4), dtype=np.float64))
        assert "N" in exc.value.errors()[0]["msg"]

    def test_triple_n_third_field_mismatch(self):
        with pytest.raises(ValidationError) as exc:
            TripleN(
                a=np.ones((5, 2), dtype=np.float32),
                b=np.ones((5, 3), dtype=np.float32),
                c=np.ones((6,), dtype=np.float32),
            )
        msg = exc.value.errors()[0]["msg"]
        assert "N" in msg
        assert "c" in msg

    def test_error_includes_bound_and_actual_sizes(self):
        with pytest.raises(ValidationError) as exc:
            VectorPair(
                a=np.ones(5, dtype=np.float64),
                b=np.ones(7, dtype=np.float64),
            )
        msg = exc.value.errors()[0]["msg"]
        assert "5" in msg
        assert "7" in msg

    def test_error_includes_field_name(self):
        with pytest.raises(ValidationError) as exc:
            MatMulPair(
                weights=np.ones((3, 4), dtype=np.float64),
                bias=np.ones((9,), dtype=np.float64),
            )
        assert "bias" in exc.value.errors()[0]["msg"]


# ---------------------------------------------------------------------------
# Interaction with per-field shape constraints
# ---------------------------------------------------------------------------

class TestPerFieldConstraintInteraction:
    def test_fixed_dim_violation_is_field_level_error(self):
        class ConstrainedPair(ShapeBindingModel):
            a: NDArray[np.float32, (1, "N")]
            b: NDArray[np.float32, ("N",)]

        with pytest.raises(ValidationError) as exc:
            ConstrainedPair(
                a=np.ones((2, 5), dtype=np.float32),  # dim 0 must be 1
                b=np.ones((5,), dtype=np.float32),
            )
        assert exc.value.errors()[0]["loc"] == ("a",)

    def test_symbol_consistent_across_mixed_constraints(self):
        class MixedPair(ShapeBindingModel):
            a: NDArray[np.float32, (1, "N")]
            b: NDArray[np.float32, ("N",)]

        m = MixedPair(
            a=np.ones((1, 8), dtype=np.float32),
            b=np.ones((8,), dtype=np.float32),
        )
        assert m.a.shape == (1, 8)
        assert m.b.shape == (8,)

    def test_symbol_mismatch_across_mixed_constraints(self):
        class MixedPair(ShapeBindingModel):
            a: NDArray[np.float32, (1, "N")]
            b: NDArray[np.float32, ("N",)]

        with pytest.raises(ValidationError) as exc:
            MixedPair(
                a=np.ones((1, 8), dtype=np.float32),
                b=np.ones((5,), dtype=np.float32),  # N=8 from a, got 5
            )
        assert "N" in exc.value.errors()[0]["msg"]
