# numpy-type-annotations

Pydantic v2-compatible type annotations for NumPy arrays. Supports dtype validation, shape validation, and round-trip JSON serialization.

---

## Usage

```python
import numpy as np
from pydantic import BaseModel
from src.ndarray import NDArray

# Any array, no constraints
class Unconstrained(BaseModel):
    arr: NDArray

# Dtype-constrained
class Float64(BaseModel):
    arr: NDArray[np.float64]

# Dtype + shape
class Matrix(BaseModel):
    weights: NDArray[np.float64, (128, 64)]

# Symbolic dimensions — "N" must resolve to the same size across both dims
class SquareMatrix(BaseModel):
    matrix: NDArray[np.float64, ("N", "N")]

# Wildcard dims — accepts any 2D shape
class AnyMatrix(BaseModel):
    arr: NDArray[np.float64, (None, None)]
```

---

## Shape Constraints

Each element of a shape tuple is one of:

| Type | Behaviour |
|------|-----------|
| `int` | Exact size required — `(3, 4)` means exactly 3 rows and 4 columns |
| `str` | Named symbolic dim — all dims with the same name must be the same size within a field |
| `None` | Wildcard — any size accepted |

### Cross-field symbol binding with `ShapeBindingModel`

Symbolic dims are enforced **within** a single field by default. To enforce consistency **across** fields, inherit from `ShapeBindingModel`:

```python
from src.models import ShapeBindingModel

class MatMulPair(ShapeBindingModel):
    weights: NDArray[np.float64, ("M", "N")]
    bias: NDArray[np.float64, ("N",)]  # N must match weights dim 1

MatMulPair(
    weights=np.ones((3, 4), dtype=np.float64),
    bias=np.ones(4, dtype=np.float64),   # OK
)

MatMulPair(
    weights=np.ones((3, 4), dtype=np.float64),
    bias=np.ones(5, dtype=np.float64),   # ValidationError: N bound to 4, got 5
)
```

---

## Serialization

Arrays serialize to a JSON object with three fields:

```json
{
  "data": "<base64-encoded raw bytes>",
  "dtype": "<f8",
  "shape": [3, 4]
}
```

- `data` — `tobytes()` output, base64-encoded. Preserves full binary precision.
- `dtype` — NumPy `dtype.str` format (e.g. `"<f8"`, `"<f4"`, `"<c16"`). The leading character encodes endianness (`<` little, `>` big, `|` not applicable), ensuring correct deserialization across platforms.
- `shape` — list of integers.

Dtype validation is strict in both Python and JSON modes — the stored dtype must exactly match the field's declared dtype. There is no implicit casting.

---

## Error messages

Validation errors identify the specific failure:

```
# Wrong type
Expected np.ndarray, got list

# Wrong dtype
Expected dtype float64, got float32

# Wrong number of dimensions
Expected 2D array, got 1D

# Wrong fixed dimension
Dim 0: expected 3, got 5

# Symbolic dimension conflict (within-field)
Dim 1: symbol 'N' bound to 4, got 7
```

---

## Repo structure

```
src/
  ndarray.py      # NDArray type annotation and validator/serializer logic
  models.py       # ShapeBindingModel for cross-field symbol binding

tests/
  test_ndarray.py # NDArray unit tests (validation, serialization, round-trip, edge cases)
  test_models.py  # ShapeBindingModel unit tests
```

---

## Design choices

**Why Pydantic v2 custom type instead of a validator?**
Using `__get_pydantic_core_schema__` integrates directly with Pydantic's core validation pipeline. The type annotation carries its constraints (`_dtype`, `_shape`) as class attributes on a dynamically-created subclass, so the same `NDArray` class works as a type hint at every call site without any extra boilerplate.

**Why base64 over JSON arrays of numbers?**
Base64-encoding `tobytes()` output preserves the exact binary representation of every dtype (including `float32`, `complex128`, `uint8`, `bool`). A JSON number array would lose precision for `float32` (JSON has no 32-bit float type), cannot represent complex numbers, and is substantially larger on the wire. The ideal solution would be a raw bytes + metadata format (like `.npy`), but that requires abandoning JSON as the transport — base64 is the best option within that constraint.

**Why `dtype.str` instead of `dtype.name` in the wire format?**
`dtype.name` (`"float64"`) loses endianness information. `dtype.str` (`"<f8"`) encodes byte order, which matters when deserializing on a different platform. NumPy's `frombuffer` uses the stored string directly to interpret raw bytes, so this prevents silent data corruption on big-endian systems.

**Why always strict dtype validation?**
Coercive casting between dtypes (e.g. `float32` → `float64`) is lossy in one direction and silently wrong in the other (`complex128` → `float64` discards imaginary parts without error). For scientific data there is no safe general casting rule, so mismatches are always rejected. The producer is responsible for serializing with the correct dtype.

**Why `ShapeBindingModel` as a separate class?**
Per-field symbolic dims (`("N", "N")`) bind within a single field, so a square matrix constraint doesn't need cross-field awareness. Cross-field binding requires a `model_validator` that runs after all fields are populated — putting this in a separate base class keeps the single-field and multi-field concerns cleanly separated.

---

## What would be implemented with more time

**Lazy / chunked array support for large shapes**

The current base64 approach requires the full array in memory during serialization. For the stated requirement of shapes up to `(10_000, 200, 200, 200)` at `float32` (~150 GB), this is not viable.

The intended approach is a `LazyNDArray` companion class backed by [Zarr](https://zarr.readthedocs.io):

- Arrays are stored in a chunked Zarr store (local directory or cloud object storage via `fsspec`)
- The Pydantic model serializes to a JSON reference `{"store": "<path>", "key": "data", "dtype": ..., "shape": ...}` rather than embedding the binary blob
- Deserialization returns a `zarr.Array` handle — reads are lazy and chunk-aware, so `arr[0:100]` fetches only the relevant chunks
- Zarr supports per-chunk compression (Blosc/Zstd), reducing storage footprint significantly

```python
# Proposed API
class LargeModel(BaseModel):
    data: LazyNDArray[np.float32, (10_000, 200, 200, 200)]

m = LargeModel.model_validate_json(json_str)
chunk = m.data[0:10]   # reads ~10/10_000ths of the data from disk
```

**Compression for the existing base64 path**

For moderate array sizes, adding zlib or lz4 compression before base64 encoding reduces wire size by 2–10× for typical scientific data without changing the public API. A `compressed` flag in the wire format would distinguish compressed from uncompressed payloads.
