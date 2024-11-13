from random import Random

import numpy as np

from ..interpreter import DSL, Symbol, eval_program, form_to_sentence


def search_random_pop(
    forms: list, env: dict, grid_true: np.ndarray, rng: Random
) -> DSL:
    best_form = None
    best_acc = -1.0

    while forms:
        idx = rng.randrange(len(forms))
        f = forms.pop(idx)
        try:
            p = [[Symbol("apply"), Symbol("make-grid"), *grid_true.shape, f]]
            grid_pred = eval_program(p, env={**env}).draw()
            acc = _grid_accuracy(grid_true, grid_pred)

            if acc > best_acc:
                best_acc = acc
                best_form = p[0]  # Get form out of program

            if acc == 1.0:
                break
        except:
            continue

    if best_form is None:
        return ";; No working form found", best_acc

    return form_to_sentence(best_form), best_acc


def _grid_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    if true.shape != pred.shape:
        return -1.0
    return float((true == pred).sum() / true.size)
