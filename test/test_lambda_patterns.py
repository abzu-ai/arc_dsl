from arc_dsl.grid import DeclarativeGrid
from arc_dsl.bitmap import Bitmap
from arc_dsl.interpreter import eval_sentence, DEFAULT_ENV

from arc_dsl.search.types import ArgPattern, Pattern
from arc_dsl.search.patterns import resolve_patterns
from arc_dsl.search.lambda_pattern import get_lambda_pattern


def test_lambda_bitmap_color_comparison():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (!= (get-bitmap-color var1) GREY))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=Bitmap)], type=bool
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_bitmap_area_lambda():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (* (get-bitmap-h var1) (get-bitmap-w var1)))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=Bitmap)], type=int
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_simple_addition_lambda():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (+ var1 var2))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=int), ArgPattern(type=int)], type=int
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_multiplication_lambda():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (* var1 var2))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=int), ArgPattern(type=int)], type=int
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_less_than_comparison_lambda():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (< var1 var2))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=int), ArgPattern(type=int)], type=bool
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_using_map_with_nested_lambda():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (map (lambda (x) (+ x 1)) var1))"
    expected_output = None  # The function should give up due to nested Lambda
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_using_sort():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (sort < var1))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=list)], type=list
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_calls_unknown_function():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (unknown-function var1))"
    expected_output = None  # The function should give up due to unknown function call
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_with_unused_parameter():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (* var1 2))"
    expected_output = Pattern(
        symbol=None,
        arg_classes=[
            ArgPattern(type=int),
            ArgPattern(type=None),  # Type unknown for unused parameter
        ],
        type=int,
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_returns_constant_integer():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) 42)"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=None)], type=int
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_higher_order_lambda():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (var1 var2))"
    expected_output = (
        None  # The function should give up due to parameter used as function
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_with_nested_function_calls():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (+ (* var1 var1) 1))"
    expected_output = Pattern(symbol=None, arg_classes=[ArgPattern(type=int)], type=int)
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_using_get_bitmap_x():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (get-bitmap-x var1))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=Bitmap)], type=int
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_modifies_bitmap():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (set-bitmap-x var1 var2))"
    expected_output = Pattern(
        symbol=None,
        arg_classes=[ArgPattern(type=Bitmap), ArgPattern(type=int)],
        type=Bitmap,
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_returns_list():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (list var1 var2))"
    expected_output = None  # The function should give up due to 'LIST' being avoided
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_using_make_grid():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2 var3) (make-grid var1 var2 var3))"
    expected_output = Pattern(
        symbol=None,
        arg_classes=[
            ArgPattern(type=int),
            ArgPattern(type=int),
            ArgPattern(type=Bitmap),
        ],
        type=DeclarativeGrid,
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_ignores_parameter():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) 0)"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=None)], type=int
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_uses_eval():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (eval var1))"
    expected_output = None  # The function should give up due to use of 'eval'
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_multiple_nested_calls():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (+ (* var1 var1) (* var2 var2)))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=int), ArgPattern(type=int)], type=int
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_returns_boolean():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1) (< var1 10))"
    expected_output = Pattern(
        symbol=None, arg_classes=[ArgPattern(type=int)], type=bool
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output


def test_lambda_parameter_as_function():
    default_patterns = resolve_patterns(DEFAULT_ENV)
    lisp_code = "(lambda (var1 var2) (var1 var2))"
    expected_output = (
        None  # The function should give up due to parameter used as function
    )
    l = eval_sentence(lisp_code)
    actual_output = get_lambda_pattern(l, default_patterns)
    assert actual_output == expected_output
