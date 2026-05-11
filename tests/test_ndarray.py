"""Tests for the NDArray type annotation."""

import base64
import json

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from src.ndarray import NDArray


# ---------------------------------------------------------------------------
# Helper models used across multiple test classes
# ---------------------------------------------------------------------------

class AnyModel(BaseModel):
    arr: NDArray

class Float64Model(BaseModel):
    arr: NDArray[np.float64]

class Float32Model(BaseModel):
    arr: NDArray[np.float32]

class Shape3Model(BaseModel):
    arr: NDArray[np.float64, (3,)]

class Shape2x4Model(BaseModel):
    arr: NDArray[np.float64, (2, 4)]

class SymbolicModel(BaseModel):
    arr: NDArray[np.float64, ("N",)]

class WildcardModel(BaseModel):
    arr: NDArray[np.float64, (None, None)]

class MixedModel(BaseModel):
    arr: NDArray[np.float32, (1, "N")]

class Complex128Model(BaseModel):
    arr: NDArray[np.complex128]

class SquareMatrixModel(BaseModel):
    matrix: NDArray[np.float64, ("N", "N")]


# ---------------------------------------------------------------------------
# Python-mode validation
# ---------------------------------------------------------------------------

class TestPythonValidation:
    def test_any_ndarray_accepted(self):
        m = AnyModel(arr=np.array([1, 2, 3]))
        assert m.arr.tolist() == [1, 2, 3]

    def test_list_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnyModel(arr=[1, 2, 3])
        assert "Expected np.ndarray" in exc.value.errors()[0]["msg"]

    def test_dict_rejected(self):
        with pytest.raises(ValidationError):
            AnyModel(arr={"a": 1})

    def test_int_rejected(self):
        with pytest.raises(ValidationError):
            AnyModel(arr=42)

    def test_correct_dtype_accepted(self):
        m = Float64Model(arr=np.array([1.0], dtype=np.float64))
        assert m.arr.dtype == np.float64

    def test_wrong_dtype_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Float64Model(arr=np.array([1.0], dtype=np.float32))
        assert "Expected dtype" in exc.value.errors()[0]["msg"]

    def test_dtype_error_includes_expected_and_actual(self):
        with pytest.raises(ValidationError) as exc:
            Float64Model(arr=np.array([1.0], dtype=np.float32))
        msg = exc.value.errors()[0]["msg"]
        assert "float64" in msg
        assert "float32" in msg

    def test_correct_1d_shape_accepted(self):
        m = Shape3Model(arr=np.ones(3, dtype=np.float64))
        assert m.arr.shape == (3,)

    def test_wrong_dim_size_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Shape3Model(arr=np.ones(5, dtype=np.float64))
        assert "Dim" in exc.value.errors()[0]["msg"]

    def test_wrong_ndim_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Shape3Model(arr=np.ones((3, 3), dtype=np.float64))
        assert "D array" in exc.value.errors()[0]["msg"]

    def test_correct_2d_shape_accepted(self):
        m = Shape2x4Model(arr=np.ones((2, 4), dtype=np.float64))
        assert m.arr.shape == (2, 4)

    def test_wrong_first_dim_rejected(self):
        with pytest.raises(ValidationError):
            Shape2x4Model(arr=np.ones((3, 4), dtype=np.float64))

    def test_wrong_second_dim_rejected(self):
        with pytest.raises(ValidationError):
            Shape2x4Model(arr=np.ones((2, 5), dtype=np.float64))

    def test_invalid_shape_constraint_type_raises(self):
        class BadModel(BaseModel):
            arr: NDArray[np.float64, (3.5,)]
        with pytest.raises(TypeError, match="Invalid shape constraint at dim 0"):
            BadModel(arr=np.ones(1, dtype=np.float64))

    def test_symbolic_dim_accepts_any_size(self):
        for n in [1, 5, 100]:
            m = SymbolicModel(arr=np.ones(n, dtype=np.float64))
            assert m.arr.shape == (n,)

    def test_square_matrix_accepted(self):
        m = SquareMatrixModel(matrix=np.eye(4, dtype=np.float64))
        assert m.matrix.shape == (4, 4)

    def test_square_matrix_various_sizes_accepted(self):
        for n in [1, 3, 10, 100]:
            m = SquareMatrixModel(matrix=np.eye(n, dtype=np.float64))
            assert m.matrix.shape == (n, n)

    def test_square_matrix_mismatch_rejected(self):
        with pytest.raises(ValidationError) as exc:
            SquareMatrixModel(matrix=np.ones((3, 4), dtype=np.float64))
        msg = exc.value.errors()[0]["msg"]
        assert "N" in msg
        assert "3" in msg
        assert "4" in msg

    def test_square_matrix_error_identifies_dimensions(self):
        with pytest.raises(ValidationError) as exc:
            SquareMatrixModel(matrix=np.ones((5, 9), dtype=np.float64))
        msg = exc.value.errors()[0]["msg"]
        assert "5" in msg
        assert "9" in msg

    def test_square_matrix_symbols_not_shared_across_instantiations(self):
        SquareMatrixModel(matrix=np.eye(3, dtype=np.float64))
        m = SquareMatrixModel(matrix=np.eye(5, dtype=np.float64))
        assert m.matrix.shape == (5, 5)

    def test_symbolic_shape_and_dtype_both_validated(self):
        class EmbeddingMatrix(BaseModel):
            weights: NDArray[np.float32, ("vocab", "dim")]

        m = EmbeddingMatrix(weights=np.ones((512, 64), dtype=np.float32))
        assert m.weights.shape == (512, 64)
        assert m.weights.dtype == np.float32

        with pytest.raises(ValidationError) as exc:
            EmbeddingMatrix(weights=np.ones((512, 64), dtype=np.float64))
        assert "Expected dtype" in exc.value.errors()[0]["msg"]

        with pytest.raises(ValidationError) as exc:
            EmbeddingMatrix(weights=np.ones(512, dtype=np.float32))
        assert "D array" in exc.value.errors()[0]["msg"]

    def test_wildcard_dims_accept_any_shape(self):
        for shape in [(1, 1), (3, 5), (100, 200)]:
            m = WildcardModel(arr=np.ones(shape, dtype=np.float64))
            assert m.arr.shape == shape

    def test_mixed_fixed_and_symbolic_accepted(self):
        m = MixedModel(arr=np.ones((1, 128), dtype=np.float32))
        assert m.arr.shape == (1, 128)

    def test_mixed_fixed_dim_violated_rejected(self):
        with pytest.raises(ValidationError):
            MixedModel(arr=np.ones((2, 128), dtype=np.float32))

    def test_complex128_accepted(self):
        arr = np.array([1+2j, 3+4j], dtype=np.complex128)
        m = Complex128Model(arr=arr)
        assert m.arr.dtype == np.complex128

    def test_complex128_wrong_dtype_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Complex128Model(arr=np.array([1+2j], dtype=np.complex64))
        assert "Expected dtype" in exc.value.errors()[0]["msg"]

    def test_scalar_0d_accepted(self):
        class ScalarModel(BaseModel):
            val: NDArray[np.int64]
        m = ScalarModel(val=np.array(42, dtype=np.int64))
        assert m.val.shape == ()
        assert int(m.val) == 42


# ---------------------------------------------------------------------------
# __class_getitem__ parameterisation
# ---------------------------------------------------------------------------

class TestClassGetitem:
    def test_dtype_only(self):
        T = NDArray[np.float64]
        assert T._dtype == np.dtype(np.float64)
        assert T._shape is None

    def test_dtype_and_shape(self):
        T = NDArray[np.float32, (2, 3)]
        assert T._dtype == np.dtype(np.float32)
        assert T._shape == (2, 3)

    def test_symbolic_shape_stored(self):
        T = NDArray[np.float64, ("N", "M")]
        assert T._shape == ("N", "M")

    def test_none_shape_stored(self):
        T = NDArray[np.float64, (None, None)]
        assert T._shape == (None, None)

    def test_too_many_params_raises_type_error(self):
        with pytest.raises(TypeError):
            NDArray[np.float64, (3,), np.float32]

    def test_base_class_unconstrained(self):
        assert NDArray._dtype is None
        assert NDArray._shape is None

    def test_parameterisation_does_not_mutate_base(self):
        NDArray[np.float64]
        assert NDArray._dtype is None
        assert NDArray._shape is None


# ---------------------------------------------------------------------------
# Serialization wire format
# ---------------------------------------------------------------------------

class TestSerializationFormat:
    def test_output_has_exactly_three_keys(self):
        m = Float64Model(arr=np.array([1.0, 2.0], dtype=np.float64))
        payload = json.loads(m.model_dump_json())
        assert set(payload["arr"].keys()) == {"data", "dtype", "shape"}

    def test_data_field_is_valid_base64(self):
        m = Float64Model(arr=np.array([1.0], dtype=np.float64))
        payload = json.loads(m.model_dump_json())
        base64.b64decode(payload["arr"]["data"])  # must not raise

    def test_dtype_field_matches_array(self):
        m = Float32Model(arr=np.array([1.0], dtype=np.float32))
        payload = json.loads(m.model_dump_json())
        assert payload["arr"]["dtype"] == np.dtype(np.float32).str

    def test_dtype_field_includes_endianness_marker(self):
        m = Float64Model(arr=np.array([1.0], dtype=np.float64))
        payload = json.loads(m.model_dump_json())
        dtype_str = payload["arr"]["dtype"]
        assert dtype_str[0] in ("<", ">", "|"), f"Expected endianness prefix, got {dtype_str!r}"

    def test_shape_field_matches_array(self):
        m = Shape2x4Model(arr=np.ones((2, 4), dtype=np.float64))
        payload = json.loads(m.model_dump_json())
        assert payload["arr"]["shape"] == [2, 4]

    def test_1d_shape_is_single_element_list(self):
        m = AnyModel(arr=np.ones(7))
        payload = json.loads(m.model_dump_json())
        assert payload["arr"]["shape"] == [7]

    def test_raw_bytes_match_tobytes(self):
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        m = Float64Model(arr=arr)
        payload = json.loads(m.model_dump_json())
        assert base64.b64decode(payload["arr"]["data"]) == arr.tobytes()

    def test_complex128_dtype_in_wire_format(self):
        m = Complex128Model(arr=np.array([1+2j], dtype=np.complex128))
        payload = json.loads(m.model_dump_json())
        assert payload["arr"]["dtype"] == np.dtype(np.complex128).str

    def test_complex128_bytes_encode_real_and_imaginary(self):
        arr = np.array([1+2j], dtype=np.complex128)
        m = Complex128Model(arr=arr)
        payload = json.loads(m.model_dump_json())
        assert base64.b64decode(payload["arr"]["data"]) == arr.tobytes()


# ---------------------------------------------------------------------------
# Round-trip (Python → JSON → Python)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_1d_float64_values_and_dtype(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        restored = Float64Model.model_validate_json(Float64Model(arr=arr).model_dump_json())
        assert np.array_equal(arr, restored.arr)
        assert restored.arr.dtype == np.float64

    def test_1d_float32_values_and_dtype(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        restored = Float32Model.model_validate_json(Float32Model(arr=arr).model_dump_json())
        assert np.array_equal(arr, restored.arr)
        assert restored.arr.dtype == np.float32

    def test_2d_shape_preserved(self):
        arr = np.arange(8, dtype=np.float64).reshape(2, 4)
        restored = Shape2x4Model.model_validate_json(Shape2x4Model(arr=arr).model_dump_json())
        assert restored.arr.shape == (2, 4)
        assert np.array_equal(arr, restored.arr)

    def test_3d_array(self):
        class ThreeDModel(BaseModel):
            arr: NDArray[np.float64]
        arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        restored = ThreeDModel.model_validate_json(ThreeDModel(arr=arr).model_dump_json())
        assert restored.arr.shape == (2, 3, 4)
        assert np.array_equal(arr, restored.arr)

    def test_integer_dtype_preserved(self):
        class IntModel(BaseModel):
            arr: NDArray[np.int32]
        arr = np.array([1, -2, 3], dtype=np.int32)
        restored = IntModel.model_validate_json(IntModel(arr=arr).model_dump_json())
        assert np.array_equal(arr, restored.arr)
        assert restored.arr.dtype == np.int32

    def test_float_values_preserved_exactly(self):
        arr = np.array([1.23456789, -9.87654321, 0.0], dtype=np.float64)
        restored = Float64Model.model_validate_json(Float64Model(arr=arr).model_dump_json())
        np.testing.assert_array_equal(arr, restored.arr)

    def test_restored_array_is_writable(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        restored = Float64Model.model_validate_json(Float64Model(arr=arr).model_dump_json())
        restored.arr[0] = 99.0  # frombuffer returns read-only without .copy()

    def test_unconstrained_json_round_trip(self):
        arr = np.array([1.0, 2.0, 3.0])
        restored = AnyModel.model_validate_json(AnyModel(arr=arr).model_dump_json())
        np.testing.assert_array_equal(arr, restored.arr)

    def test_shape_constraint_checked_after_deserialize(self):
        class Unconstrained(BaseModel):
            arr: NDArray[np.float64]
        wrong = Unconstrained(arr=np.ones(5, dtype=np.float64))
        with pytest.raises(ValidationError):
            Shape3Model.model_validate_json(wrong.model_dump_json())

    def test_square_matrix_round_trip(self):
        arr = np.arange(9, dtype=np.float64).reshape(3, 3)
        restored = SquareMatrixModel.model_validate_json(
            SquareMatrixModel(matrix=arr).model_dump_json()
        )
        assert restored.matrix.shape == (3, 3)
        assert np.array_equal(arr, restored.matrix)

    def test_complex128_round_trip(self):
        arr = np.array([1+2j, -3+4.5j, 0+0j], dtype=np.complex128)
        restored = Complex128Model.model_validate_json(
            Complex128Model(arr=arr).model_dump_json()
        )
        assert restored.arr.dtype == np.complex128
        np.testing.assert_array_equal(arr, restored.arr)

    def test_complex64_round_trip(self):
        class Complex64Model(BaseModel):
            arr: NDArray[np.complex64]
        arr = np.array([1+2j, 3+4j], dtype=np.complex64)
        restored = Complex64Model.model_validate_json(Complex64Model(arr=arr).model_dump_json())
        assert restored.arr.dtype == np.complex64
        np.testing.assert_array_equal(arr, restored.arr)

    def test_bool_dtype_round_trip(self):
        class BoolModel(BaseModel):
            mask: NDArray[np.bool_]
        arr = np.array([True, False, True], dtype=np.bool_)
        restored = BoolModel.model_validate_json(BoolModel(mask=arr).model_dump_json())
        assert restored.mask.dtype == np.bool_
        np.testing.assert_array_equal(restored.mask, arr)

    def test_uint8_dtype_round_trip(self):
        class PixelModel(BaseModel):
            pixels: NDArray[np.uint8, (100,)]
        arr = np.arange(100, dtype=np.uint8)
        restored = PixelModel.model_validate_json(PixelModel(pixels=arr).model_dump_json())
        assert restored.pixels.dtype == np.uint8
        np.testing.assert_array_equal(restored.pixels, arr)

    def test_int32_and_int64_round_trip(self):
        class IntPairModel(BaseModel):
            small: NDArray[np.int32]
            large: NDArray[np.int64]
        small = np.array([1, 2, 3], dtype=np.int32)
        large = np.array([1, 2, 3], dtype=np.int64)
        restored = IntPairModel.model_validate_json(
            IntPairModel(small=small, large=large).model_dump_json()
        )
        assert restored.small.dtype == np.int32
        assert restored.large.dtype == np.int64

    def test_empty_1d_round_trip(self):
        arr = np.array([], dtype=np.float64)
        restored = Float64Model.model_validate_json(Float64Model(arr=arr).model_dump_json())
        assert restored.arr.shape == (0,)
        assert restored.arr.size == 0

    def test_empty_2d_round_trip(self):
        class EmptyRowsModel(BaseModel):
            arr: NDArray[np.int32, (0, 5)]
        arr = np.zeros((0, 5), dtype=np.int32)
        restored = EmptyRowsModel.model_validate_json(EmptyRowsModel(arr=arr).model_dump_json())
        assert restored.arr.shape == (0, 5)

    def test_scalar_0d_round_trip(self):
        class ScalarModel(BaseModel):
            val: NDArray[np.float64]
        val = np.array(3.14, dtype=np.float64)
        restored = ScalarModel.model_validate_json(ScalarModel(val=val).model_dump_json())
        assert restored.val.shape == ()
        assert float(restored.val) == pytest.approx(3.14)

    def test_fortran_order_values_preserved(self):
        class F32_3x4Model(BaseModel):
            arr: NDArray[np.float32, (3, 4)]
        f_arr = np.asfortranarray(np.arange(12, dtype=np.float32).reshape((3, 4)))
        assert f_arr.flags["F_CONTIGUOUS"]
        restored = F32_3x4Model.model_validate_json(F32_3x4Model(arr=f_arr).model_dump_json())
        np.testing.assert_array_equal(restored.arr, f_arr)

    def test_non_contiguous_slice_round_trip(self):
        class Vec5Model(BaseModel):
            arr: NDArray[np.float64, (5,)]
        sliced = np.arange(10, dtype=np.float64)[::2]  # non-contiguous
        assert not sliced.flags["C_CONTIGUOUS"]
        restored = Vec5Model.model_validate_json(Vec5Model(arr=sliced).model_dump_json())
        np.testing.assert_array_equal(restored.arr, sliced)

    def test_large_array_round_trip(self):
        class LargeModel(BaseModel):
            data: NDArray[np.float32, (100, 100)]
        arr = np.random.randn(100, 100).astype(np.float32)
        restored = LargeModel.model_validate_json(LargeModel(data=arr).model_dump_json())
        np.testing.assert_array_almost_equal(restored.data, arr)


# ---------------------------------------------------------------------------
# JSON deserialization error cases
# ---------------------------------------------------------------------------

class TestDeserializationErrors:
    def test_list_payload_rejected(self):
        with pytest.raises(ValidationError) as exc:
            Float64Model.model_validate_json('{"arr": [1.0, 2.0, 3.0]}')
        assert "Expected object with" in exc.value.errors()[0]["msg"]

    def test_missing_data_key_rejected(self):
        bad = json.dumps({"arr": {"dtype": "float64", "shape": [3]}})
        with pytest.raises(ValidationError) as exc:
            Float64Model.model_validate_json(bad)
        assert "Expected object with" in exc.value.errors()[0]["msg"]

    def test_missing_shape_key_rejected(self):
        arr = np.ones(3, dtype=np.float64)
        bad = json.dumps({"arr": {
            "data": base64.b64encode(arr.tobytes()).decode(),
            "dtype": "float64",
        }})
        with pytest.raises(ValidationError) as exc:
            Float64Model.model_validate_json(bad)
        assert "Expected object with" in exc.value.errors()[0]["msg"]

    def test_corrupted_base64_rejected(self):
        bad = json.dumps({"arr": {"data": "!!!invalid!!!", "dtype": "float64", "shape": [3]}})
        with pytest.raises(ValidationError) as exc:
            Float64Model.model_validate_json(bad)
        assert "Cannot deserialize" in exc.value.errors()[0]["msg"]

    def test_bad_dtype_string_rejected(self):
        arr = np.ones(3, dtype=np.float64)
        bad = json.dumps({"arr": {
            "data": base64.b64encode(arr.tobytes()).decode(),
            "dtype": "not_a_real_dtype",
            "shape": [3],
        }})
        with pytest.raises(ValidationError) as exc:
            Float64Model.model_validate_json(bad)
        assert "Cannot deserialize" in exc.value.errors()[0]["msg"]

    def test_dtype_mismatch_rejected(self):
        class F32Source(BaseModel):
            arr: NDArray[np.float32]
        src = F32Source(arr=np.array([1.0, 2.0], dtype=np.float32))
        with pytest.raises(ValidationError) as exc:
            Float64Model.model_validate_json(src.model_dump_json())
        msg = exc.value.errors()[0]["msg"]
        assert "float64" in msg
        assert "float32" in msg

    def test_shape_size_inconsistent_with_data(self):
        arr = np.ones(3, dtype=np.float64)  # 3 elements = 24 bytes
        bad = json.dumps({"arr": {
            "data": base64.b64encode(arr.tobytes()).decode(),
            "dtype": "float64",
            "shape": [5],  # claims 5 elements but data only has 3
        }})
        with pytest.raises(ValidationError) as exc:
            Float64Model.model_validate_json(bad)
        assert "Cannot deserialize" in exc.value.errors()[0]["msg"]


# ---------------------------------------------------------------------------
# JSON schema generation
# ---------------------------------------------------------------------------

def _resolve(model_schema: dict, field: str) -> dict:
    """Follow $ref for a field schema if needed."""
    s = model_schema["properties"][field]
    if "$ref" in s:
        key = s["$ref"].split("/")[-1]
        return model_schema["$defs"][key]
    return s


class TestJsonSchema:
    def test_type_is_object(self):
        s = _resolve(AnyModel.model_json_schema(), "arr")
        assert s["type"] == "object"

    def test_required_properties_present(self):
        s = _resolve(AnyModel.model_json_schema(), "arr")
        for key in ("data", "dtype", "shape"):
            assert key in s["properties"]
        assert set(s["required"]) == {"data", "dtype", "shape"}

    def test_data_property_has_base64_encoding(self):
        s = _resolve(AnyModel.model_json_schema(), "arr")
        assert s["properties"]["data"].get("contentEncoding") == "base64"

    def test_x_dtype_present_when_constrained(self):
        s = _resolve(Float64Model.model_json_schema(), "arr")
        assert s.get("x-dtype") == "float64"

    def test_x_dtype_absent_when_unconstrained(self):
        s = _resolve(AnyModel.model_json_schema(), "arr")
        assert "x-dtype" not in s

    def test_x_shape_present_when_constrained(self):
        s = _resolve(Shape2x4Model.model_json_schema(), "arr")
        assert s.get("x-shape") == [2, 4]

    def test_none_in_shape_becomes_question_mark(self):
        class NoneShapeModel(BaseModel):
            arr: NDArray[np.float64, (None, 3)]
        s = _resolve(NoneShapeModel.model_json_schema(), "arr")
        assert s["x-shape"] == ["?", 3]

    def test_symbolic_dim_preserved_in_schema(self):
        s = _resolve(SymbolicModel.model_json_schema(), "arr")
        assert s["x-shape"] == ["N"]
