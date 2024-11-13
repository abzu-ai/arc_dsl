# Code for running the ARC-prize submission notebook
import numpy as np

from arc_dsl.data import ARCTask, create_task_examples
from arc_dsl.solve import solve_task


def parse_raw_tasks_dict(raw_task_input: dict):
    return [
        ARCTask(
            id=task_id,
            train=create_task_examples(task_id, task_data, "train"),
            test=create_task_examples(task_id, task_data, "test"),
        )
        for task_id, task_data in raw_task_input.items()
    ]


def solve_tasks(raw_task_input: dict):
    tasks = parse_raw_tasks_dict(raw_task_input)
    solutions = {}

    for task in tasks:
        solution = solve_task(task)
        if solution is None:
            solution = [
                [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])]
                for _ in task.test
            ]

        solutions[task.id] = [
            {f"attempt_{i+1}": output.tolist() for i, output in enumerate(test_output)}
            for test_output in solution
        ]
    return solutions
