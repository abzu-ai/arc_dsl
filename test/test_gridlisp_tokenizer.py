import pytest
from arc_dsl.models.tokenizer import GridlispTokenizer
from arc_dsl.interpreter import eval_sentence
from arc_dsl.errors import ParseError

from arc_dsl.models.tokenizer import (
    COMPLETION_TOKEN,
    END_TOKEN,
    START_TOKEN,
    GridlispTokenizer,
)
from arc_dsl.interpreter import eval_sentence

from .gridlisp_examples import gridlisp_examples


@pytest.fixture
def tokenizer():
    return GridlispTokenizer()


def test_canonical_tokenization(tokenizer):
    s1 = tokenizer.make_tokenizable("(+ 2 3)")
    s2 = tokenizer.make_tokenizable(" (+ 2 3)")
    s3 = tokenizer.make_tokenizable("( + 2 3)")
    s4 = tokenizer.make_tokenizable(" ( + 2 3)")
    assert s1 == s2 == s3 == s4 == " ( + 2 3)"


def get_test_examples():
    """Get DSL examples from test_dsl_eval"""
    dsl_examples = []
    for dsl, out in gridlisp_examples():

        if isinstance(out, type) and issubclass(out, Exception):
            continue

        dsl_examples.append((dsl, out))
    return dsl_examples


@pytest.mark.parametrize("dsl,expected", get_test_examples())
def test_tokenizer_roundtrip(tokenizer, dsl, expected):
    """Test that encoding and decoding preserves the DSL string and evaluation"""
    dsl_round_trip = tokenizer.decode(tokenizer.encode(dsl))

    # Check that decoded string evaluates to same output
    out_round_trip = eval_sentence(dsl_round_trip)

    assert (
        out_round_trip == expected
    ), f"Decoded DSL {dsl} should produce same output, not {out_round_trip}"


def test_invalid_token(tokenizer):
    # No way to convert this into valid tokenizable DSL
    with pytest.raises(ParseError):
        tokenizer.encode("invalid_token")


def test_invalid_id(tokenizer):
    """Test that invalid token IDs raise ValueError"""
    with pytest.raises(ValueError, match="Unknown token ID"):
        tokenizer.decode([999])


def test_init():
    """Test that the tokenizer can be initialized with a custom vocabulary"""
    vocab = [START_TOKEN, END_TOKEN, COMPLETION_TOKEN, "a", "b", "c"]
    id_to_token = {i: token for i, token in enumerate(vocab)}
    tokenizer = GridlispTokenizer(id_to_token)

    assert tokenizer.id_to_token == id_to_token
    assert len(tokenizer) == len(vocab)


def test_save_load(tmp_path):
    """Test saving and loading the tokenizer"""
    tokenizer = GridlispTokenizer()

    # Save the tokenizer to a file path
    file_path = tmp_path / "tokenizer.json"
    tokenizer.save(file_path)

    # Load the tokenizer from the file
    loaded_tokenizer = GridlispTokenizer.load(file_path)

    # Verify that the loaded tokenizer has the same state
    assert tokenizer.id_to_token == loaded_tokenizer.id_to_token
    assert tokenizer.token_to_id == loaded_tokenizer.token_to_id
    assert tokenizer.pattern.pattern == loaded_tokenizer.pattern.pattern
