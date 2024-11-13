import numpy as np
import tensorflow as tf

from arc_dsl import ARCError
from arc_dsl.interpreter import eval_program, parse_sentence


def masked_loss(label, pred):
    """Make absolutely sure we ignore padding, even if the ._keras_mask is not flowing all the way to the logits."""
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,  # from_logits=on/off depends on the activation from the last dense layer
        ignore_class=0,  # Ignore padding
    )
    return loss_object(label, pred)


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0
    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def _matrix_acc(x: np.ndarray, y: np.ndarray) -> float:
    total_pixils = max(x.shape[0], y.shape[0]) * max(x.shape[1], y.shape[1])

    if total_pixils == 0:
        return np.float32(0.0)

    overlap_rows, overlap_cols = min(x.shape[0], y.shape[0]), min(
        x.shape[1], y.shape[1]
    )
    correct_pixils = np.sum(
        x[:overlap_rows, :overlap_cols] == y[:overlap_rows, :overlap_cols]
    )

    return correct_pixils / total_pixils


def pixil_acc_grids(x_dsl: str, y_dsl: str):
    try:
        x_grid: np.ndarray = eval_program(parse_sentence(x_dsl)).draw()
        y_grid: np.ndarray = eval_program(parse_sentence(y_dsl)).draw()
    except ARCError:
        return np.float32(0.0)

    return _matrix_acc(x_grid, y_grid)


def pixil_acc_bitmaps(x_dsl: str, y_dsl: str):
    x_dsl = f"(make-grid 30 30 {x_dsl})"
    y_dsl = f"(make-grid 30 30 {y_dsl})"

    return pixil_acc_grids(x_dsl, y_dsl)
