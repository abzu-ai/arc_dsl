import numpy as np

from arc_dsl.errors import ParseError, EvalError
from arc_dsl.bitmap import Bitmap
from arc_dsl.interpreter import Symbol, DEFAULT_ENV


def gridlisp_examples():
    return [
        (
            """
            (setparam var1 (make-grid 5 3))
            (get-grid-h var1)
            """,
            5,
        ),
        (
            """
            (setparam var1 (make-grid 5 3))
            (get-grid-w var1)
            """,
            3,
        ),
        ("(% 7 3)", 1),
        ("(% 8 4)", 0),
        ("(/ 6 2)", 3),
        ("(/ 7 2)", 3),
        ("(!= 3 3)", False),
        ("(!= 3 4)", True),
        ("(= 3 3)", True),
        ("(= 3 4)", False),
        ("(filter (lambda (var1) (= var1 2)) '(1 2 3 2))", [2, 2]),
        (
            """
            (setparam var1 3)
            (setparam var2 2)
            (setparam var3 '(var1 var2))
            (apply * var3)
            """,
            EvalError,  # var3 is a list of symbols, not values!
        ),
        ("(sort < '(3 1 2))", [1, 2, 3]),
        (
            """
            (setparam var1 3)
            VAR1
            """,
            3,
        ),
        (
            """
            (setparam var1 3) ;; I am a comment!
            ;; I am also a comment!
            var1
            """,
            3,
        ),
        (
            """
            (setparam var1
                (lambda
                    ()
                    (+ var2 var3)))

            (setparam var2 2)
            (setparam var3 3)

            (var1)
            """,
            5,
        ),
        (
            """
            (setparam var1
                (lambda
                    (var1)
                    var1))

            (var1 2)
            """,
            2,
        ),
        (
            """
            (setparam var1
                (lambda
                    (var2 var3)
                    (+ var2 var3)))

            (setparam var2 2)
            (setparam var3 3)

            (var1 1 2)
            """,  # Local scope overrides global
            3,
        ),
        (
            """
            (setparam var1
                (lambda
                    (var3)
                    (+ var2 var3)))

            (setparam var2 2)

            (var1 1)
            """,  # Python dict pointer makes this work
            3,
        ),
        (
            """
            (setparam var1
                (lambda
                    (var2 var3)
                    (* (+ var2 var3) 2)))

            (var1 1 2)
            """,
            6,
        ),
        ("(setparam var1 10) (setparam var2 (+ var1 2)) var2", 12),
        ("(setparam var1 '(1 2 3)) (apply + var1)", 6),
        ("(setparam var1 '(1 2 3)) var1", [1, 2, 3]),
        ("(setparam var1 10) (setparam var1 5) var1", 5),
        ("(setparam var1 10) var1", 10),
        ("(* 2 2) (+ 3 3)", 6),  # Get output of last form
        ("apply", DEFAULT_ENV["APPLY"]),
        ("3", 3),
        ("'3", 3),
        ("'()", []),
        ("(1 2 3)", EvalError),  # int object is not callable
        ("'(1 '2 3)", [1, [Symbol("quote"), 2], 3]),
        (
            "'(12 3 45)",
            [12, 3, 14],
        ),  # 45 is too large for grids, parsed to 45 % 30 = 15
        ("'(1 2 3 '(4 5))", [1, 2, 3, [Symbol("quote"), [4, 5]]]),
        ("'(1 2 3 (+ 4 5))", [1, 2, 3, [Symbol("+"), 4, 5]]),
        ("'+", Symbol("+")),
        ("(+ 1 2)", 3),
        ("(+ 1 2 3)", 6),
        ("(+ 1 2", ParseError),
        ("(* (+ 1 2) 3)", 9),
        ("(* (+ 12) 3)", 36),
        ("(map + '(1 2))", [1, 2]),
        ("(map + '(1 2) '(3 2))", [4, 4]),
        ("(map + '(1 2 3) '(3 2 1))", [4, 4, 4]),
        ("(map + '(1 2 3) '(3 2))", [4, 4]),
        ("(make-bitmap 0 1 2 2 3 10)", Bitmap(0, 1, 2, 2, 3, [10])),
        ("(make-grid 5 5)", np.zeros((5, 5), dtype=int)),
        ("'(+ 2 3)", [Symbol("+"), 2, 3]),
        ("(eval 5)", 5),
        ("(eval '(+ 2 3))", 5),
        ("(eval (+ 2 3))", 5),  # Equivalent to (eval 5)
        ("(eval '(* 2 3))", 6),
        ("(eval '(+ 1 2 3 4))", 10),
        ("(eval ''5)", 5),
        ("(eval ''(1 2 3))", [1, 2, 3]),
        ("(eval '(eval '(+ 2 3)))", 5),
        ("(eval '(map + '(1 2) '(3 4)))", [4, 6]),
        ("(eval '(eval '(* 2 (+ 3 4))))", 14),
        ("(apply + '(2 3))", 5),
        ("(apply + '(1 2 3))", 6),
        ("(apply * '(2 3 4))", 24),
        ("(apply + '())", 0),
        ("(apply * '())", 1),
        ("(apply + '(1))", 1),
        ("(apply * '(5))", 5),
        ("(apply + 1 2 '(5 2))", 10),
        ("(list 1 2 3)", [1, 2, 3]),
        (
            "(make-grid 5 5 (make-bitmap 0 1 2 2 GREEN 9))",
            np.array(
                [
                    [0, 3, 0, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            "(make-grid 5 5 (set-bitmap-color (make-bitmap 0 1 2 2 GREEN 9) 2))",
            np.array(
                [
                    [0, 2, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            "(make-grid 4 5 (make-bitmap 0 1 2 2 GREEN 9) (make-bitmap 1 2 2 2 BLUE 11))",
            np.array(
                [
                    [0, 3, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            "(make-grid 5 5 (make-bitmap 3 3 3 3 BLUE 15 15 8))",
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                ]
            ),
        ),
        (
            """
            (map get-bitmap-color
                (list
                    (make-bitmap 0 1 2 2 GREEN 9)
                    (make-bitmap 1 2 2 2 BLUE 11)))
            """,
            [3, 1],
        ),
        (
            "(map get-bitmap-color (get-grid-bitmaps (make-grid 4 5 (make-bitmap 0 1 2 2 GREEN 9) (make-bitmap 1 2 2 2 BLUE 11))))",
            [3, 1],
        ),
    ]
