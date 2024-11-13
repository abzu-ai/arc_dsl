import numpy as np

from arc_dsl.data import ARCTask, ARCTaskExample, TRAIN_FOLDER


def test_arctaskexample_mapping():
    """Test that ARCTaskExample behaves like a mapping."""
    input_array = np.array([[1, 2], [3, 4]])
    output_array = np.array([[4, 3], [2, 1]])
    example = ARCTaskExample(id="abc/train/1", input=input_array, output=output_array)

    # Test __getitem__
    assert np.array_equal(example["input"], input_array)
    assert np.array_equal(example["output"], output_array)


def test_arctaskexample_equality_and_hash():
    """Test that ARCTaskExample uses id for hashing and equality."""
    input = np.array([[1, 2], [3, 4]])
    output = np.array([[4, 3], [2, 1]])

    example1 = ARCTaskExample(id="example1", input=input, output=output)
    example2 = ARCTaskExample(id="example1", input=input, output=output)
    example3 = ARCTaskExample(id="example3", input=input, output=output)

    # Examples with the same id should be equal
    assert example1 == example2
    assert hash(example1) == hash(example2)

    # Examples with different ids should not be equal
    assert example1 != example3
    assert hash(example1) != hash(example3)


def test_arctask_parses_arc_files():
    """Test that a arc problem can be loaded"""
    problem_file = TRAIN_FOLDER / "67385a82.json"

    # Read the problem using _read_problem
    task = ARCTask.from_file(problem_file)

    assert task.id == "67385a82"
    assert len(task.train) == 4
    assert len(task.test) == 1

    # Check example ids get assigned sequentially
    assert [example.id for example in task.train] == [
        "67385a82/train/0",
        "67385a82/train/1",
        "67385a82/train/2",
        "67385a82/train/3",
    ]

    assert [example.id for example in task.test] == ["67385a82/test/0"]

    # Check train example
    train_example = task.train[0]
    assert np.array_equal(
        train_example.input, np.array([[3, 3, 0], [0, 3, 0], [3, 0, 3]])
    )

    assert np.array_equal(
        train_example.output, np.array([[8, 8, 0], [0, 8, 0], [3, 0, 3]])
    )

    # Check test example
    test_example = task.test[0]
    assert np.array_equal(
        test_example.input,
        np.array(
            [
                [3, 0, 3, 0, 3],
                [3, 3, 3, 0, 0],
                [0, 0, 0, 0, 3],
                [0, 3, 3, 0, 0],
                [0, 3, 3, 0, 0],
            ]
        ),
    )
    assert np.array_equal(
        test_example.output,
        np.array(
            [
                [8, 0, 8, 0, 3],
                [8, 8, 8, 0, 0],
                [0, 0, 0, 0, 3],
                [0, 8, 8, 0, 0],
                [0, 8, 8, 0, 0],
            ]
        ),
    )
