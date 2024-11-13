from arc_dsl.interpreter import Symbol, Lambda
from arc_dsl.grid import DeclarativeGrid
from arc_dsl.bitmap import Bitmap

from .types import Pattern, ArgPattern
from .lambda_pattern import get_lambda_pattern


def resolve_patterns(env: dict[Symbol, any]) -> list[Pattern]:
    patterns = []

    if Symbol("+") in env:
        patterns.append(
            Pattern(Symbol("+"), [ArgPattern(type=int), ArgPattern(type=int)], int)
        )
    if Symbol("*") in env:
        patterns.append(
            Pattern(Symbol("*"), [ArgPattern(type=int), ArgPattern(type=int)], int)
        )
    if Symbol("<") in env:
        patterns.append(
            Pattern(Symbol("<"), [ArgPattern(type=int), ArgPattern(type=int)], bool)
        )
    if Symbol("=") in env:
        patterns.append(
            Pattern(Symbol("="), [ArgPattern(type=int), ArgPattern(type=int)], bool)
        )
    if Symbol("!=") in env:
        patterns.append(
            Pattern(Symbol("!="), [ArgPattern(type=int), ArgPattern(type=int)], bool)
        )
    if Symbol("%") in env:
        patterns.append(
            Pattern(Symbol("%"), [ArgPattern(type=int), ArgPattern(type=int)], int)
        )
    if Symbol("/") in env:
        patterns.append(
            Pattern(Symbol("/"), [ArgPattern(type=int), ArgPattern(type=int)], int)
        )

    if Symbol("SORT") in env:
        patterns.append(
            Pattern(
                Symbol("SORT"),
                [ArgPattern(func=True, arity=2), ArgPattern(type=list)],
                list,
            )
        )

    if Symbol("MAP") in env:
        patterns.append(
            Pattern(
                Symbol("MAP"),
                [ArgPattern(func=True, arity=1), ArgPattern(type=list)],
                list,
            )
        )
        patterns.append(
            Pattern(
                Symbol("MAP"),
                [
                    ArgPattern(func=True, arity=2),
                    ArgPattern(type=list),
                    ArgPattern(type=list),
                ],
                list,
            )
        )

    # avoid these funny ones for now
    # if Symbol("APPLY") in env:
    #    patterns.append(Pattern(Symbol("APPLY"), [ArgPattern(func=True), ArgPattern()]))

    # if Symbol("LIST") in env:
    #    patterns.append(Pattern(Symbol("LIST"), [ArgPattern(), ArgPattern()]))

    if Symbol("DECOMPOSITION") in env:
        patterns.append(
            Pattern(Symbol("DECOMPOSITION"), type=DeclarativeGrid),
        )

    if Symbol("MAKE-GRID") in env:
        patterns.append(
            Pattern(
                Symbol("MAKE-GRID"),
                [ArgPattern(type=int), ArgPattern(type=int), ArgPattern(type=Bitmap)],
                DeclarativeGrid,
            )
        )
    if Symbol("GET-GRID-BITMAPS") in env:
        patterns.append(
            Pattern(
                Symbol("GET-GRID-BITMAPS"), [ArgPattern(type=DeclarativeGrid)], list
            )
        )
    if Symbol("MAKE-BITMAP") in env:
        patterns.append(
            Pattern(
                Symbol("MAKE-BITMAP"),
                [
                    ArgPattern(type=int),  # x
                    ArgPattern(type=int),  # y
                    ArgPattern(type=int),  # h
                    ArgPattern(type=int),  # w
                    ArgPattern(type=int),  # color
                    ArgPattern(),  # encoding
                ],
                Bitmap,
            )
        )

    if Symbol("GET-GRID-H") in env:
        patterns.append(
            Pattern(Symbol("GET-GRID-H"), [ArgPattern(type=DeclarativeGrid)], int)
        )
    if Symbol("GET-GRID-W") in env:
        patterns.append(
            Pattern(Symbol("GET-GRID-W"), [ArgPattern(type=DeclarativeGrid)], int)
        )

    if Symbol("GET-BITMAP-X") in env:
        patterns.append(Pattern(Symbol("GET-BITMAP-X"), [ArgPattern(type=Bitmap)], int))
    if Symbol("GET-BITMAP-Y") in env:
        patterns.append(Pattern(Symbol("GET-BITMAP-Y"), [ArgPattern(type=Bitmap)], int))
    if Symbol("GET-BITMAP-H") in env:
        patterns.append(Pattern(Symbol("GET-BITMAP-H"), [ArgPattern(type=Bitmap)], int))
    if Symbol("GET-BITMAP-W") in env:
        patterns.append(Pattern(Symbol("GET-BITMAP-W"), [ArgPattern(type=Bitmap)], int))
    if Symbol("GET-BITMAP-COLOR") in env:
        patterns.append(
            Pattern(Symbol("GET-BITMAP-COLOR"), [ArgPattern(type=Bitmap)], int)
        )

    if Symbol("SET-BITMAP-X") in env:
        patterns.append(
            Pattern(
                Symbol("SET-BITMAP-X"),
                [ArgPattern(type=Bitmap), ArgPattern(type=int)],
                Bitmap,
            )
        )
    if Symbol("SET-BITMAP-Y") in env:
        patterns.append(
            Pattern(
                Symbol("SET-BITMAP-Y"),
                [ArgPattern(type=Bitmap), ArgPattern(type=int)],
                Bitmap,
            )
        )
    if Symbol("SET-BITMAP-H") in env:
        patterns.append(
            Pattern(
                Symbol("SET-BITMAP-H"),
                [ArgPattern(type=Bitmap), ArgPattern(type=int)],
                Bitmap,
            )
        )
    if Symbol("SET-BITMAP-W") in env:
        patterns.append(
            Pattern(
                Symbol("SET-BITMAP-W"),
                [ArgPattern(type=Bitmap), ArgPattern(type=int)],
                Bitmap,
            )
        )
    if Symbol("SET-BITMAP-COLOR") in env:
        patterns.append(
            Pattern(
                Symbol("SET-BITMAP-COLOR"),
                [ArgPattern(type=Bitmap), ArgPattern(type=int)],
                Bitmap,
            )
        )

    for var_name, value in env.items():
        if value is None:
            continue
        if isinstance(var_name, Symbol) and var_name.startswith("VAR"):
            if isinstance(value, Lambda):
                # NOTE: One would want to check this in order such that lambdas that reference builtins are type-inferred before lambdas that use other lambdas.
                lambda_pattern = get_lambda_pattern(value, patterns)
                if lambda_pattern is not None:
                    patterns.append(
                        Pattern(
                            var_name, lambda_pattern.arg_classes, lambda_pattern.type
                        )
                    )
            else:
                patterns.append(Pattern(var_name, type=type(value)))

    return patterns
