from arc_dsl.interpreter import form_to_sentence, Symbol

from arc_dsl.search.enumerate import enumerate_forms
from arc_dsl.search.patterns import Pattern, ArgPattern


def test_map_not_repeated():
    patterns = [
        Pattern(
            Symbol("MAP"),
            [ArgPattern(func=True, arity=1), ArgPattern(type=list)],
            list,
        ),
        Pattern(Symbol("f"), type=int, arg_classes=[ArgPattern(type=int)]),
        Pattern(Symbol("data"), type=list),
    ]

    forms = enumerate_forms(patterns, max_depth=2)
    sentences = set(map(form_to_sentence, forms))

    # But should still generate valid forms like: (map f data)
    assert "(MAP F DATA)" in sentences

    # Should not generate nested maps like: (map f (map f data))
    assert "(MAP F (MAP F DATA))" not in sentences


def test_enumerate_forms_expected_type():
    all_patterns = [
        Pattern(Symbol("a"), type=int),
        Pattern(Symbol("b"), type=str),
        Pattern(Symbol("c"), type=bool),
        Pattern(Symbol("d"), type=int, arg_classes=[ArgPattern(type=bool)]),
    ]

    int_forms = enumerate_forms(all_patterns, expected_type=int)
    assert set(map(form_to_sentence, int_forms)) == {"A", "(D C)"}

    str_forms = enumerate_forms(all_patterns, expected_type=str)
    assert set(map(form_to_sentence, str_forms)) == {"B"}

    bool_forms = enumerate_forms(all_patterns, expected_type=bool)
    assert set(map(form_to_sentence, bool_forms)) == {"C"}

    all_forms = enumerate_forms(all_patterns)
    assert set(map(form_to_sentence, all_forms)) == {
        "A",
        "B",
        "C",
        "(D C)",
    }
