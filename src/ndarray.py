"""Pydantic v2-compatible annotated NDArray type."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

# Each element of a ShapeConstraint is one of:
#   int  — exact dimension size required
#   str  — named symbolic dim; same name must resolve to the same size
#   None — anonymous wildcard, any size accepted
ShapeConstraint = tuple[int | str | None, ...]

# ---------------------------------------------------------------------------
# Validator logic
# ---------------------------------------------------------------------------

# Validator for dtype constraints; raises if the array's dtype doesn't match the expected one.
def _validate_dtype(v: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if v.dtype != dtype:
        raise PydanticCustomError(
            "ndarray_dtype",
            "Expected dtype {expected}, got {actual}",
            {"expected": dtype.name, "actual": v.dtype.name},
        )
    return v

def _validate_shape(v: np.ndarray, shape: ShapeConstraint) -> np.ndarray:
    # Check if the number of dimensions matches
    if v.ndim != len(shape):
        raise PydanticCustomError(
            "ndarray_ndim",
            "Expected {expected}D array, got {actual}D",
            {"expected": len(shape), "actual": v.ndim},
        )

    # Check each dimension against the corresponding constraint
        # i.e. for shape=(3, "N", None), check dim 0 == 3, bind dim 1 to "N", accept any size for dim 2
    for i, (exp, act) in enumerate(zip(shape, v.shape)):
        if isinstance(exp, int):
            if act != exp:
                raise PydanticCustomError(
                    "ndarray_shape",
                    "Dim {dim}: expected {expected}, got {actual}",
                    {"dim": i, "expected": exp, "actual": act},
                )
        # If dim in constraint is a str or None, represents a symbolic dimension/no preference. MVP will ignore the dimension for initial validation
        else:
            continue
    
    return v
        # None means any size is accepted, so no check needed


# def _validate_shape(v: np.ndarray, shape: ShapeConstraint) -> np.ndarray:
#     if v.ndim != len(shape):
#         raise PydanticCustomError(
#             "ndarray_ndim",
#             "Expected {expected}D array, got {actual}D",
#             {"expected": len(shape), "actual": v.ndim},
#         )

#     bindings: dict[str, int] = {}
#     for i, (actual_size, constraint) in enumerate(zip(v.shape, shape)):
#         if constraint is None:
#             continue
#         if isinstance(constraint, int):
#             if actual_size != constraint:
#                 raise PydanticCustomError(
#                     "ndarray_shape",
#                     "Dim {dim}: expected {expected}, got {actual}",
#                     {"dim": i, "expected": constraint, "actual": actual_size},
#                 )
#         elif isinstance(constraint, str):
#             if constraint in bindings:
#                 if actual_size != bindings[constraint]:
#                     raise PydanticCustomError(
#                         "ndarray_shape_symbol",
#                         "Dim {dim}: symbol '{symbol}' bound to {bound}, got {actual}",
#                         {
#                             "dim": i,
#                             "symbol": constraint,
#                             "bound": bindings[constraint],
#                             "actual": actual_size,
#                         },
#                     )
#             else:
#                 bindings[constraint] = actual_size

#     return v


# ---------------------------------------------------------------------------
# Serializer logic
# ---------------------------------------------------------------------------


def _serialize_ndarray(v: np.ndarray) -> dict:
    return {
        "data": base64.b64encode(v.tobytes()).decode("ascii"),
        "dtype": v.dtype.name,
        "shape": list(v.shape),
    }


# ---------------------------------------------------------------------------
# Type annotation class
# ---------------------------------------------------------------------------


if TYPE_CHECKING:
    # Stub so type-checkers accept NDArray[dtype, shape] subscript syntax without
    # errors and treat plain NDArray as np.ndarray for attribute access.
    class NDArray(np.ndarray):  # type: ignore[type-arg]
        @classmethod
        def __class_getitem__(cls, _: Any) -> Any: ...  # type: ignore[override]
else:
    class NDArray:
        """Pydantic-compatible NumPy array type annotation.

        Never instantiated directly — used purely as a type hint.

        Usage:
            x: NDArray                     # any array, any dtype/shape
            x: NDArray[np.float64]        # dtype-constrained
            x: NDArray[np.float64, (3, "N")]  # dtype + shape-constrained
            x: NDArray[np.float64, (None, None)]  # dtype-constrained, any 2D shape
        """

        _dtype: np.dtype | None = None
        _shape: ShapeConstraint | None = None

        def __class_getitem__(cls, params: Any) -> type:
            if not isinstance(params, tuple):
                params = (params,)

            # Quick sanity check to catch common param structure mistakes like NDArray[np.float64, (3, 3), "extra"]        
            if isinstance(params, tuple) and len(params) > 2:
                raise TypeError("NDArray[...] accepts at most 2 parameters: dtype and shape")

            dtype: np.dtype | None = None
            shape: ShapeConstraint | None = None
            for p in params:
                if isinstance(p, tuple):
                    shape = p
                else:
                    dtype = np.dtype(p)

            return type(
                f"NDArray[{getattr(dtype, 'name', dtype)}, {shape}]", #New class name
                (cls,), #Inherit from NDArray
                {"_dtype": dtype, "_shape": shape}, #Set clas attributes for dtype and shape
            )

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            dtype = cls._dtype
            shape = cls._shape

            def validate_python(v: Any) -> np.ndarray:
                if not isinstance(v, np.ndarray):
                    raise PydanticCustomError(
                        "ndarray_type",
                        "Expected np.ndarray, got {type_name}",
                        {"type_name": type(v).__name__},
                    )
                if dtype is not None:
                    _validate_dtype(v, dtype)
                if shape is not None:
                    _validate_shape(v, shape)
                return v

            def validate_json(v: Any) -> np.ndarray:
                if not isinstance(v, dict) or not {"data", "dtype", "shape"}.issubset(v):
                    raise PydanticCustomError(
                        "ndarray_format",
                        "Expected object with 'data', 'dtype', and 'shape' keys",
                        {},
                    )
                try:
                    raw = base64.b64decode(v["data"])
                    stored_dtype = np.dtype(v["dtype"])
                    stored_shape = tuple(v["shape"])
                    arr = np.frombuffer(raw, dtype=stored_dtype).reshape(stored_shape).copy()
                except Exception as exc:
                    raise PydanticCustomError(
                        "ndarray_convert",
                        "Cannot deserialize ndarray: {error}",
                        {"error": str(exc)},
                    ) from exc
                if dtype is not None and arr.dtype != dtype:
                    arr = arr.astype(dtype)
                if shape is not None:
                    _validate_shape(arr, shape)
                return arr

            return core_schema.json_or_python_schema(
                json_schema=core_schema.no_info_plain_validator_function(validate_json),
                python_schema=core_schema.no_info_plain_validator_function(validate_python),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    _serialize_ndarray,
                    info_arg=False,
                ),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, _core_schema: CoreSchema, _handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            schema: JsonSchemaValue = {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "contentEncoding": "base64"},
                    "dtype": {"type": "string"},
                    "shape": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["data", "dtype", "shape"],
            }
            if cls._dtype is not None:
                schema["x-dtype"] = cls._dtype.name
            if cls._shape is not None:
                schema["x-shape"] = [s if s is not None else "?" for s in cls._shape]
            return schema
    