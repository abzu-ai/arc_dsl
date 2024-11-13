import pytest
import numpy as np

from arc_dsl.interpreter import parse_sentence, eval_program

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
