from dataclasses import dataclass
from random import Random
import logging

import humanize

from arc_dsl.grid import DeclarativeGrid
from arc_dsl.models.decomposition.model import DecompositionModel

from .data import ARCTask, ARCTaskExample
from .interpreter import DSL, DEFAULT_ENV, Symbol, eval_sentence
from .search import search_random_pop
from .search.enumerate import enumerate_forms
from .search.patterns import resolve_patterns

logger = logging.getLogger(__name__)

decomposition_model = DecompositionModel()


def decompose(task: ARCTask) -> dict[ARCTaskExample, DeclarativeGrid]:
    result = {}

    examples = task.train + task.test
    input_grids = [example.input for example in examples]
    decompositions = decomposition_model.decompose_batch(input_grids)

    for example, decomposition in zip(examples, decompositions):
        result[example] = decomposition

    return result


@dataclass
class ARCSolution:
    ideas: DSL
    grid_transform: DSL
    accuracy: float


def solve_task(task: ARCTask, n_tries=1, rng: Random = None):
    logger.info(f"Attempting to solve task {task.id}..")
    rng = rng or Random()
    decompositions = decompose(task)

    # Try to solve each training example
    for i in range(n_tries):
        logger.info(f"Attempt no {i+1}..")
        example_idx = i % len(task.train)
        example = task.train[example_idx]

        solution = _find_candidate(example, decompositions[example], rng)
        if solution.accuracy == 1:
            n_solved = _check_solves_train(solution, task, decompositions)
            logger.info(f"Solution solved {n_solved}/{len(task.train)} train examples.")
            if n_solved == len(task.train):
                return [
                    [
                        _eval_solution(solution, decompositions[test_example])
                        for _ in range(2)
                    ]
                    for test_example in task.test
                ]

    logger.info("No perfect solution found")
    return None


def _find_candidate(
    example: ARCTaskExample,
    decomposition: DeclarativeGrid,
    rng: Random,
) -> ARCSolution:
    # Get some good ideas
    ideas = _ideate(example, None)

    # Set up environment for search
    preamble_env = _make_env(decomposition, ideas)

    ## Enumerate possible forms
    patterns = resolve_patterns(preamble_env)
    all_forms = enumerate_forms(
        patterns,
        max_depth=3,
        expected_type=list,  # Expecting list of bitmaps as output
    )
    logger.info(f"Enumerated {humanize.intword(len(all_forms))} forms.")

    grid_transform, acc = search_random_pop(
        all_forms, preamble_env, example.output, rng
    )
    if acc == 1:
        logger.info("Found explicit solution!")
    else:
        logger.info(f"No explicit solution, best accuracy is {acc:.2f}")

    return ARCSolution(ideas, grid_transform, acc)


def _ideate(example: ARCTaskExample, decomposition: DSL) -> DSL:
    # Hardcoded to ea32f347 (andorra)
    return """
    (setparam var1 (list RED YELLOW BLUE))
    (setparam var2
      (lambda
        (var1)
        (* (get-bitmap-h var1) (get-bitmap-w var1))))

    (setparam var3
      (lambda
        (var4 var5)
        (< (var2 var4) (var2 var5))))
    """


def _make_env(decomposition: DeclarativeGrid, ideas: DSL) -> dict:
    env = {**DEFAULT_ENV}
    env[Symbol("DECOMPOSITION")] = decomposition

    eval_sentence(ideas, env=env)
    return env


def _check_solves_train(
    candidate: ARCSolution,
    task: ARCTask,
    decompositions: dict[ARCTaskExample, DeclarativeGrid],
):
    n_solved = 0
    for example in task.train:
        decomposition = decompositions[example]
        output = _eval_solution(candidate, decomposition)
        if output.shape == example.output.shape and (output == example.output).all():
            n_solved += 1
    return n_solved


def _eval_solution(solution: ARCSolution, decomposition: DeclarativeGrid):
    preamble_env = _make_env(decomposition, solution.ideas)

    try:
        output_grid = eval_sentence(solution.grid_transform, env=preamble_env)
        assert isinstance(output_grid, DeclarativeGrid)
        return output_grid.draw()
    except:
        return None
