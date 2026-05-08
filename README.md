# numpy-type-annotations

thought process:

Essential pydantic functionality
- dtype validation:
- shape validation:
- Serialization 
    - ndarray -> JSON -> ndarray

Numpy ndarray characteristics:
- shape(n,n): Defines shape of the numpy entry in terms of rows, columns, etc. 
- dtype(): Defines datatype of entries within the ndarray 
- buffer?

To address:

Extending from pydantic functionality, why doesn't it work currently?

Explain how pydantic functionality will be implemented

Repo structure?

User usage design

Test cases

Edge cases/failure modes

Design choices along the way

What is left? 
- Verify serialization method for larger numpy arrays
- Rewrite tests in pytest, following python standards
- Update readme with user guide, summary, and features to be added later
