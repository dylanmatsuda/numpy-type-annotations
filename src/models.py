"""Pydantic v2-compatible Model classes."""

from pydantic import BaseModel, model_validator

class ShapeBindingModel(BaseModel):
    """BaseModel subclass that supports binding shape symbols to actual dimensions.
    This allows for validating that multiple arrays with shared shape symbols have consistent dimensions.
    """
    @model_validator(mode="after")
    def validate_shape_bindings(self) -> "ShapeBindingModel":
        bindings: dict[str, int] = {}

        for field_name, field_info in self.__class__.model_fields.items():
            field_type = field_info.annotation
            if hasattr(field_type, "_shape"):
                shape = getattr(field_type, "_shape")
                value = getattr(self, field_name)

                for i, dim_constraint in enumerate(shape):
                    if isinstance(dim_constraint, str):
                        # Symbolic dimension
                        if dim_constraint in bindings:
                            if value.shape[i] != bindings[dim_constraint]:
                                raise ValueError(
                                    f"Field '{field_name}', dim {i}: symbol '{dim_constraint}' bound to {bindings[dim_constraint]}, got {value.shape[i]}"
                                )
                        else:
                            bindings[dim_constraint] = value.shape[i]

        return self
        