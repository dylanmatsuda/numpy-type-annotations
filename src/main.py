"""Usage examples for the AnnotatedArray type annotation."""

import numpy as np
from pydantic import BaseModel, ValidationError

from ndarray import NDArray
from models import ShapeBindingModel

class AnyArray(BaseModel):
    data: NDArray


class TypedArray(BaseModel):
    weights: NDArray[np.float64]


class ConstrainedArray(BaseModel):
    # First dim fixed at 1, second dim symbolic ("N" = any size)
    features: NDArray[np.float32, (1, "N")]


class SquareMatrix(BaseModel):
    # Both dims share the same symbol, so they must be equal
    matrix: NDArray[np.float64, ("N", "N")]


# weights (M, N) and bias (N,) share symbol "N" — must match
class MatMulPair(ShapeBindingModel):
    weights: NDArray[np.float64, ("M", "N")]
    bias: NDArray[np.float64, ("N",)]
    

# Three fields sharing "N" — all must agree
class TripleN(ShapeBindingModel):
    a: NDArray[np.float32, ("N", 2)]
    b: NDArray[np.float32, ("N", 3)]
    c: NDArray[np.float32, ("N",)]


if __name__ == "__main__":
    import json

    # --- Base64 serialization: inspect wire format ---
    m = TypedArray(weights=np.array([1.0, 2.0, 3.0], dtype=np.float64))
    json_str = m.model_dump_json()
    payload = json.loads(json_str)
    print("Serialized keys :", list(payload["weights"].keys()))
    print("Stored dtype    :", payload["weights"]["dtype"])
    print("Stored shape    :", payload["weights"]["shape"])

    # --- Round-trip: values and dtype preserved exactly ---
    original = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    restored = TypedArray.model_validate_json(json_str)
    print("Values match    :", np.array_equal(original, restored.weights))
    print("Dtype match     :", original.dtype == restored.weights.dtype)

    # --- 2D constrained array round-trip ---
    feat = np.ones((1, 64), dtype=np.float32)
    c = ConstrainedArray(features=feat)
    c_restored = ConstrainedArray.model_validate_json(c.model_dump_json())
    print("2D round-trip   :", np.array_equal(feat, c_restored.features), c_restored.features.shape)

    # --- Non-trivial values preserved across reshape ---
    class GridModel(BaseModel):
        grid: NDArray[np.float64, (3, 4)]

    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    grid_restored = GridModel.model_validate_json(GridModel(grid=arr).model_dump_json())
    print("Exact values    :", np.array_equal(arr, grid_restored.grid))

    # --- Error: list payload (old format) rejected ---
    try:
        TypedArray.model_validate_json('{"weights": [1.0, 2.0, 3.0]}')
    except ValidationError as e:
        print("Bad format err  :", e.errors()[0]["msg"])

    # --- Error: corrupted base64 data field ---
    bad = json.dumps({"weights": {"data": "!!!not-base64!!!", "dtype": "float64", "shape": [3]}})
    try:
        TypedArray.model_validate_json(bad)
    except ValidationError as e:
        print("Bad base64 err  :", e.errors()[0]["msg"])

    # --- Error: stored shape violates field constraint ---
    class AnyFloat32(BaseModel):
        features: NDArray[np.float32]

    wrong_shape = AnyFloat32(features=np.ones((2, 4), dtype=np.float32))
    try:
        ConstrainedArray.model_validate_json(wrong_shape.model_dump_json())
    except ValidationError as e:
        print("Shape mismatch  :", e.errors()[0]["msg"])

    # --- Shape + dtype constraints ---
    ok = ConstrainedArray(features=np.ones((1, 128), dtype=np.float32))
    print("Shape OK  :", ok.features.shape)

    # --- Symbol consistency ---
    sq = SquareMatrix(matrix=np.eye(4, dtype=np.float64))
    print("Square OK :", sq.matrix.shape)

    # --- Validation errors ---
    try:
        ConstrainedArray(features=np.ones((2, 128), dtype=np.float32))
    except ValidationError as e:
        print("Shape err :", e.errors()[0]["msg"])

    try:
        TypedArray(weights=np.array([1.0], dtype=np.float32))
    except ValidationError as e:
        print("Dtype err :", e.errors()[0]["msg"])

    try:
        SquareMatrix(matrix=np.ones((3, 4), dtype=np.float64))
    except ValidationError as e:
        print("Symbol err:", e.errors()[0]["msg"])

    # --- ShapeBindingModel: cross-field symbol consistency ---

    # OK: N=4 in both fields
    pair = MatMulPair(
        weights=np.ones((3, 4), dtype=np.float64),
        bias=np.ones((4,), dtype=np.float64),
    )
    print("Cross-field OK :", pair.weights.shape, pair.bias.shape)

    # Fail: N bound to 4 by weights, but bias has size 5
    try:
        MatMulPair(
            weights=np.ones((3, 4), dtype=np.float64),
            bias=np.ones((5,), dtype=np.float64),
        )
    except ValidationError as e:
        print("Cross-field err:", e.errors()[0]["msg"])

    ok3 = TripleN(
        a=np.ones((5, 2), dtype=np.float32),
        b=np.ones((5, 3), dtype=np.float32),
        c=np.ones((5,), dtype=np.float32),
    )
    print("Triple symbol OK:", ok3.a.shape, ok3.b.shape, ok3.c.shape)

    try:
        TripleN(
            a=np.ones((5, 2), dtype=np.float32),
            b=np.ones((5, 3), dtype=np.float32),
            c=np.ones((6,), dtype=np.float32),
        )
    except ValidationError as e:
        print("Triple mismatch:", e.errors()[0]["msg"])
