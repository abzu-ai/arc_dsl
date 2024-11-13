from itertools import product

from .patterns import Pattern, ArgPattern


def enumerate_forms(patterns: list[Pattern], expected_type=None, max_depth=3):
    valid_patterns = _filter_options(
        None, patterns, max_depth, ArgPattern(type=expected_type)
    )
    return _enumerate_forms(valid_patterns, patterns, max_depth)


def _enumerate_forms(
    valid_patterns: list[Pattern],
    all_patterns: list[Pattern],
    remaining_depth: int = 3,
):
    all_forms = []

    for pattern in valid_patterns:
        if pattern.arity > 0:
            next_depth = remaining_depth - 1
            arg_options = []
            next_all_patterns = [p for p in all_patterns if p.symbol != pattern.symbol]

            for arg_cls in pattern.arg_classes:
                next_valid_patterns = _filter_options(
                    pattern, next_all_patterns, next_depth, arg_cls
                )
                arg_forms = _enumerate_forms(
                    next_valid_patterns, next_all_patterns, next_depth
                )
                arg_options.append(arg_forms)

            for args in product(*arg_options):
                all_forms.append([pattern.symbol, *args])
        else:
            all_forms.append(pattern.symbol)

    return all_forms


def _matches_type(pattern: Pattern, arg_pattern: ArgPattern | None = None) -> bool:
    if arg_pattern is None or arg_pattern.type is None:
        return True
    return pattern.type is arg_pattern.type


def _filter_options(
    last_pattern: Pattern | None,
    options: list[Pattern],
    remaining_depth: int,
    arg_pattern: ArgPattern = None,
):
    if arg_pattern is not None and arg_pattern.func:
        # Expecting function pointer with matching arity and return type
        return [
            Pattern(p.symbol)
            for p in options
            if p.arity == arg_pattern.arity and _matches_type(p, arg_pattern)
        ]

    result = []
    for p in options:
        if last_pattern is not None:
            # Don't repeat list creation
            if last_pattern.symbol == p.symbol == "MAP":
                continue
            if last_pattern.symbol == p.symbol == "SORT":
                continue
        if not _matches_type(p, arg_pattern):
            continue
        # When at max depth, only return patterns with no arguments
        # since we can't expand them further
        if remaining_depth <= 0 and p.arity != 0:
            continue
        result.append(p)
    return result
