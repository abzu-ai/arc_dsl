import operator
from dataclasses import replace
from math import prod
from pathlib import Path

from lark import Lark, Token, Transformer

from .bitmap import Bitmap
from .errors import ParseError, EvalError
from .grid import DeclarativeGrid

DSL = str


class Symbol(str):
    def __new__(cls, value):
        return super().__new__(cls, value.upper())

    def __eq__(self, other):
        if isinstance(other, str):
            return self.upper() == other.upper()
        return NotImplemented  # Important to return NotImplemented

    def __ne__(self, other):
        eq_result = self.__eq__(other)
        if eq_result is NotImplemented:
            return NotImplemented
        else:
            return not eq_result

    def __hash__(self):
        return hash(self.upper())


class Lambda:
    def __init__(self, params, body, env):
        self.params = [p for p in params]
        diff = set(self.params) - set(env)
        if diff:
            raise EvalError(
                f"Attempt to create lambda with illegal argument symbols '{diff}'.\n\nOnly symbols var1, var2, .. are allowed."
            )

        self.body = body
        self.env = env

    def __call__(self, *args):
        locals = dict(zip(self.params, args))
        return _eval_form(self.body, self.env | locals)

    @property
    def arity(self) -> int:
        return len(self.params)


class Listify(Transformer):
    """Map lark trees into nested lists."""

    SYMBOL = Symbol

    def INT(self, value):
        return int(value) % 31

    def list(self, ls):
        return ls

    def program(self, forms):
        return forms

    def quote(self, form):
        return [Symbol("quote")] + form


GRAMMAR_PATH = Path(__file__).parent.parent / "arc_dsl" / "gridlisp.lark"
PARSER = Lark.open(GRAMMAR_PATH, parser="lalr", start="program", transformer=Listify())


def program_to_sentence(program: list):
    return "\n".join(form_to_sentence(f) for f in program)


def form_to_sentence(form) -> DSL:
    if isinstance(form, Symbol):
        return str(form)
    elif isinstance(form, int):
        return str(form)
    elif isinstance(form, list):
        # NB! Any list is interpreted as a call, so a program as a list of forms will not produce correct output
        inner = " ".join(form_to_sentence(f) for f in form)
        return f"({inner})"
    else:
        raise ValueError(f"Cannot convert {type(form)} to sentence")


def canonicalize(sentence: str, preserve_these: list[str] = None) -> str:
    if preserve_these is None:
        preserve_these = []

    def handler(e) -> bool:
        if e.char in preserve_these:
            e.interactive_parser.feed_token(Token("SYMBOL", e.char))
            return True  # Continue

        return False  # Re-raise

    try:
        prog = PARSER.parse(sentence, on_error=handler)
    except:
        raise ParseError()

    return program_to_sentence(prog)


def parse_sentence(s: DSL):
    try:
        return PARSER.parse(s)
    except:
        raise ParseError()


def _eval_makebitmap(x, y, h, w, color, *encoding):
    return Bitmap(x, y, h, w, color, list(encoding))


def _eval_makegrid(h, w, *bitmaps):
    return DeclarativeGrid(h, w, list(bitmaps))


def _eval_sum(*args):
    return sum(args)


def _eval_prod(*args):
    return prod(args)


def _eval_map(f, *args):
    return [f(*elems) for elems in zip(*args)]


def _eval_list(*args):
    return list(args)


def _eval_apply(f, *args):
    fun_args = []
    for a in args:
        if isinstance(a, list):
            fun_args.extend(a)
        else:
            fun_args.append(a)
    return f(*fun_args)


def _eval_sort(predicate, data):
    data = data.copy()
    for i in range(1, len(data)):
        key_item = data[i]
        j = i - 1
        while j >= 0 and not predicate(data[j], key_item):
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key_item
    return data


def _eval_filter(predicate, data):
    return [x for x in data if predicate(x)]


def _eval_form(form, env: dict):
    if isinstance(form, list):
        fun_symbol, *args = form
        # update symbol value in env
        if fun_symbol == "setparam":
            symbol, value = args
            if not symbol in env:
                raise EvalError(
                    f"Trying to set value for illegal symbol '{symbol}'.\n\nVariable names must match var1, var2, ..."
                )
            env[symbol] = _eval_form(value, env)
            return

        # define lambda
        if fun_symbol == "lambda":
            return Lambda(*args, env)

        # conditional
        if fun_symbol == "cond":
            raise NotImplementedError(
                "Predicates not yet implemented. Added here for consistency with tokenizer."
            )

        # if quote, just return what you are quoting
        if fun_symbol == "quote":
            return form[1]

        # if eval call, eval eval'd arg (read this 3 times)
        if fun_symbol == "eval":
            evald_arg = _eval_form(args[0], env)
            return _eval_form(evald_arg, env)

        # regular function call
        f, *args = map(lambda form: _eval_form(form, env), form)
        return f(*args)

    if isinstance(form, Symbol):
        v = env.get(form)
        if v is None:
            raise EvalError(f"Unknown symbol '{form}'")
        return v

    # int
    return form


def eval_program(program, env=None):
    try:
        res = None
        if env is None:
            env = {**DEFAULT_ENV}
        for form in program:
            res = _eval_form(form, env)
        return res
    except EvalError:
        # Reraise the EvalError exception
        raise
    except Exception as e:
        # Raise EvalError with the original exception in the stack trace
        raise EvalError("Program could not be evaluated") from e


def eval_sentence(s: DSL, env=None):
    prog = parse_sentence(s)
    return eval_program(prog, env=env)


ALLOWED_VARS = {Symbol(f"VAR{i}"): None for i in range(1, 21)}
DEFAULT_ENV: dict[Symbol, any] = ALLOWED_VARS | {
    Symbol("DECOMPOSITION"): None,
    Symbol("BLUE"): 1,
    Symbol("RED"): 2,
    Symbol("GREEN"): 3,
    Symbol("YELLOW"): 4,
    Symbol("GREY"): 5,
    Symbol("FUCHSIA"): 6,
    Symbol("ORANGE"): 7,
    Symbol("TEAL"): 8,
    Symbol("BROWN"): 9,
    Symbol("+"): _eval_sum,
    Symbol("*"): _eval_prod,
    Symbol("<"): operator.lt,
    Symbol("="): operator.eq,
    Symbol("FILTER"): _eval_filter,
    Symbol("SORT"): _eval_sort,
    Symbol("APPLY"): _eval_apply,
    Symbol("MAP"): _eval_map,
    Symbol("LIST"): _eval_list,
    Symbol("MAKE-GRID"): _eval_makegrid,
    Symbol("GET-GRID-BITMAPS"): lambda g: g.bitmaps,
    Symbol("MAKE-BITMAP"): _eval_makebitmap,
    Symbol("GET-BITMAP-X"): lambda bm: bm.x,
    Symbol("GET-BITMAP-Y"): lambda bm: bm.y,
    Symbol("GET-BITMAP-H"): lambda bm: bm.h,
    Symbol("GET-BITMAP-W"): lambda bm: bm.w,
    Symbol("GET-BITMAP-COLOR"): lambda bm: bm.color,
    Symbol("SET-BITMAP-X"): lambda bm, new_x: replace(bm, x=new_x),
    Symbol("SET-BITMAP-Y"): lambda bm, new_y: replace(bm, y=new_y),
    Symbol("SET-BITMAP-H"): lambda bm, new_h: replace(bm, h=new_h),
    Symbol("SET-BITMAP-W"): lambda bm, new_w: replace(bm, w=new_w),
    Symbol("SET-BITMAP-COLOR"): lambda bm, new_color: replace(bm, color=new_color),
    Symbol("BLACK"): 0,
    Symbol("/"): operator.floordiv,
    Symbol("%"): operator.mod,
    Symbol("!="): operator.ne,
    Symbol("GET-GRID-H"): lambda g: g.h,
    Symbol("GET-GRID-W"): lambda g: g.w,
}
