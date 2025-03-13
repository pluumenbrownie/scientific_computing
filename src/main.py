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
