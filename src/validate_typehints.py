"""
Type validation of inputs against function type annotations.

Notes
-----
Pydantic provides this functionality built-in. See:

    - validate_call decorator:
      https://docs.pydantic.dev/latest/concepts/validation_decorator/

    - ValidationError error:
      https://docs.pydantic.dev/latest/errors/validation_errors/

TODO: review and clean this up, tests to docstring
"""

import inspect
import typing_inspect
from functools import wraps
from typing import get_type_hints, Literal


def validate_types(func):
    """
    Decorator to validate argument types, including Optional, Literal,
    and complex container types, based on function annotations.
    Raises TypeError for mismatched types.

    Notes
    -----
    Does not handle unions of nested types with other types.
    All unioned types after the first supplied type will be ignored.
    For example, the 'int' option for `data` will be ignored:

        def test(data: list[dict[str, int | str] | int] | int | None):
          pass

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints and bind arguments
        type_hints = get_type_hints(func)
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Helper to validate a single value against a type hint
        def validate_type(value, expected_type):
            # Handle Optional (alias for X | None)
            if typing_inspect.is_optional_type(expected_type):
                if value is not None:
                    validate_type(
                        value, typing_inspect.get_args(expected_type)[0]
                    )
                return

            # Handle Union types (recursively check each type in the union)
            if typing_inspect.is_union_type(expected_type):
                union_types = typing_inspect.get_args(expected_type)
                for union_type in union_types:
                    try:
                        validate_type(value, union_type)
                        # If any of the types match, stop further checks
                        return
                    except TypeError:
                        continue
                raise TypeError(
                    f"Value {value!r} does not match any of {expected_type}."
                )

            # Handle Literal types
            if typing_inspect.is_literal_type(expected_type):
                if value not in typing_inspect.get_args(expected_type):
                    raise TypeError(
                        f"Value {value!r} is not in the allowed Literal"
                        f"values: {typing_inspect.get_args(expected_type)}."
                    )
                return

            # Handle generic types like list, dict
            if typing_inspect.is_generic_type(expected_type):
                origin = typing_inspect.get_origin(expected_type)
                args = typing_inspect.get_args(expected_type)

                # Handle list
                if origin is list:
                    if not isinstance(value, list):
                        raise TypeError(
                            f"Value {value!r} is not of type list."
                        )
                    for item in value:
                        validate_type(item, args[0])

                # Handle dict
                elif origin is dict:
                    if not isinstance(value, dict):
                        raise TypeError(
                            f"Value {value!r} is not of type dict."
                        )
                    for key, val in value.items():
                        validate_type(key, args[0])  # Validate keys
                        validate_type(val, args[1])  # Validate values

                else:
                    raise TypeError(
                        f"Unsupported generic type {expected_type}."
                    )
                return

            # Handle tuple (manual check for length and type of elements)
            if isinstance(value, tuple):
                origin = typing_inspect.get_origin(expected_type)
                if origin is tuple:
                    # Check tuple length and validate each item
                    args = typing_inspect.get_args(expected_type)
                    if len(value) != len(args):
                        raise TypeError(
                            f"Tuple {value!r} does not match the expected"
                            f"length of {len(args)}."
                        )
                    for item, item_type in zip(value, args):
                        validate_type(item, item_type)
                    return

            # Handle basic types like str, int, float, bool, etc.
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Value {value!r} is not of type {expected_type}."
                )
            return

        # Validate all arguments
        for arg_name, arg_value in bound_args.arguments.items():
            if arg_name in type_hints:
                expected_type = type_hints[arg_name]
                validate_type(arg_value, expected_type)

        return func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":

    # Example Usage
    @validate_types
    def example_function(
        data: list[dict[str, int | str] | int] | None,
        scale: float | int = 1,
        label: str | None = None,
        type: Literal['a', 'b', 'c'] = 'a',
        items: list[list[str | int] | str | int] = [['a', 'b'], ['1', 2], 3, '4'],
        dims: tuple[int, int, int] = (1, 2, 3),
    ):
        """
        Function taking nested, simple, literal, and optional types.
        """
        pass

    # Test cases
    example_function(None, scale=1, label="test label", type='c')
    example_function([{"a": 1, "b": '2'}, 3], scale=1.0, label=None, type='a')

    try:
        example_function([{"a": 1, "b": 2.5}])
    except TypeError as e:
        print(f"Error: {e}\n")
    try:
        example_function([], type='d')
    except TypeError as e:
        print(f"Error: {e}\n")

    @validate_types
    def test(data: list[dict[str, int | str] | int] | int | None):
        pass

    test(data=[1, {'2': 3}, {'3': '4'}])
    test(data=None)
    # test(data=1)
