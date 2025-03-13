

 1. Non-functional code should be pushed to a new branch.
 2. Use the black-formatter extension with auto "format on save" turned on. 
 3. Function output types should not be changed without asking. 
 4. Use type hints as much as possible.
 5. Try to keep others up to date on what you are doing and what you have completed.
 6. Please try to avoid pushing temporary images.
  - Putting any files in any folder named "local" will keep those files out of the repository.
  - Any graphs meant for the report can be put in another folder.


## Type hint examples:
```python

def typed_function(
    whole_number: int, decimal_number: float, any_number: int | float
) -> float:
    """
    Example of using simple number types in a function.

    any integer multiplied by a float returns a float.
    """
    output = whole_number * decimal_number * any_number
    return output


def multiple_type_output(
    any_number_1: int | float,
    any_number_2: int | float,
) -> int | float:
    """
    Example of multiple possible outputs in a function.

    Idealy, this should be avoided.
    """
    output = any_number_1 * any_number_2
    return output


from numpy.typing import NDArray


def complex_types(
    input_string: str,
    list_of_numbers: list[int],
    numpy_array: NDArray,
) -> tuple[int, list[str]]:
    """
    Example of advanced type hint usage and returning multiple values.
    """
    output_1 = numpy_array.shape[0]
    output_2 = [input_string for i in list_of_numbers if i > 10]
    return output_1, output_2


# Use this when an external function does not work properly with type hints:
# type: ignore
```