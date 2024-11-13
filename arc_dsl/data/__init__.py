from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
import json
import random

import numpy as np


@dataclass(eq=True, frozen=True)
class ARCTaskExample(Mapping):
    id: str
    input: np.ndarray = field(compare=False)
    output: np.ndarray = field(compare=False)

    def __iter__(self):
        return iter({"input": self.input, "output": self.output})

    def __len__(self):
        return 2  # Only input and output

    def __getitem__(self, key):
        return self.__dict__[key]


def create_task_examples(task_id: str, data: dict, key: str) -> list[ARCTaskExample]:
    return [
        ARCTaskExample(
            f"{task_id}/{key}/{i}",
            np.array(ex["input"], dtype=int),
            np.array(ex["output"], dtype=int),
        )
        for i, ex in enumerate(data[key])
    ]


@dataclass
class ARCTask:
    id: str
    train: list[ARCTaskExample]
    test: list[ARCTaskExample]

    @staticmethod
    def from_file(path: Path) -> "ARCTask":
        with open(path) as file:
            data = json.load(file)

        task_id = path.stem
        return ARCTask(
            task_id,
            create_task_examples(task_id, data, "train"),
            create_task_examples(task_id, data, "test"),
        )


def get_task(id: str) -> ARCTask:
    for task in TRAIN_DATA:
        if task.id == id:
            return task
    raise ValueError(f"Task with id {id} not found")


def rand_task() -> ARCTask:
    return random.choice(TRAIN_DATA)


TRAIN_FOLDER = Path(__file__).parent / "training"
TRAIN_DATA = [ARCTask.from_file(file) for file in TRAIN_FOLDER.iterdir()]
