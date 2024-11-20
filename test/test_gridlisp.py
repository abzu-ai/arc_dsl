import pytest
import numpy as np

from arc_dsl.errors import ParseError
from arc_dsl.interpreter import parse_sentence, eval_program, canonicalize

from .gridlisp_examples import gridlisp_examples


@pytest.mark.parametrize("sentence,expected_output", gridlisp_examples())
def test_dsl_eval(sentence, expected_output):
    try:
        program = parse_sentence(sentence)
        res = eval_program(program)
        errmsg = f'Expected {expected_output} for output of "{sentence}", got {res}'

        if isinstance(expected_output, np.ndarray):
            assert (res == expected_output).all(), errmsg
        else:
            assert res == expected_output, errmsg
    except Exception as e:
        if not isinstance(expected_output, type):
            raise e
        assert isinstance(
            e, expected_output
        ), f"Expected exception {expected_output.__name__}, but got {e.__class__.__name__}"


def test_canonicalize():
    # Capitalizes symbols
    assert canonicalize("var1") == "VAR1"

    # Strips comments
    assert canonicalize("(SETPARAM VAR1 3) ;; I am a comment!") == "(SETPARAM VAR1 3)"

    # Strips whitespace
    assert canonicalize(" ( SETPARAM VAR1 3 ) ") == "(SETPARAM VAR1 3)"

    # Separates forms with newlines for readability
    assert (
        canonicalize("(SETPARAM VAR1 3) (SETPARAM VAR2 3)")
        == "(SETPARAM VAR1 3)\n(SETPARAM VAR2 3)"
    )

    # Allows preserving non-dsl symbols
    assert (
        canonicalize("(SETPARAM VAR1 ?)", preserve_these=["?"]) == "(SETPARAM VAR1 ?)"
    )

    # Raises on unknown symbols
    with pytest.raises(ParseError):
        canonicalize("(SETPARAM VAR1 ?)", preserve_these=["!"])
