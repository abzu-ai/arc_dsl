import random

import numpy as np
import tensorflow as tf

from arc_dsl.bitmap import COLOR_NAMES
from arc_dsl.models.tokenizer import GridlispTokenizer
from arc_dsl.models.training_data import normgrid, sample_multi_bitmap_grid


def bitmap_parametrized_dsl(bm, rng, threshold=0.6) -> str:
    """
    Outputs: (make-bitmap {x} {y} {h} {w} {color} {enc[0]} {enc[1]} ... {enc[k]})

    With a ? thrown in to mask some values, useful for training data generation.
    """
    parts = [str(bm.x), str(bm.y), str(bm.h), str(bm.w), COLOR_NAMES[bm.color]]

    # Always mask at least one value
    mask_ix = rng.randint(0, len(parts) - 1)
    parts[mask_ix] = "?"

    # Optionally mask more values
    for ix in range(len(parts)):
        if rng.random() > threshold:
            parts[ix] = "?"

    if rng.random() > threshold:
        parts.append(" ¿")
    else:
        parts.append(" ".join(map(str, bm.encoding)))

    return f" ( MAKE-BITMAP {' '.join(parts)})"


class PatternCompletionDataset(tf.data.Dataset):
    def _generate(sequence_length: int, batch_size: int):
        tokenizer = GridlispTokenizer()
        rng = random.Random()

        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_targets = []

        while True:
            sample = sample_multi_bitmap_grid(rng)
            dsl = "".join([bm.to_dsl() for bm in sample.grid.bitmaps])
            paramd_dsl = "".join(
                [bitmap_parametrized_dsl(bm, rng) for bm in sample.grid.bitmaps]
            )

            tokenized = (
                [tokenizer.start_token_id]
                + tokenizer.encode(paramd_dsl)
                + [tokenizer.completion_token_id]
                + tokenizer.encode(dsl)
                + [tokenizer.end_token_id]
            )

            if len(tokenized) > sequence_length:
                continue

            input_seq = tokenized[:-1]  # Strip [end] token, preserve [start] token
            target_seq = tokenized[1:]  # Strip [start] token, preserve [end] token
            padding_length = sequence_length - len(input_seq)

            input_seq_padded = np.pad(input_seq, (0, padding_length), "constant")
            target_seq_padded = np.pad(target_seq, (0, padding_length), "constant")

            # Collect samples into batches
            batch_encoder_inputs.append(normgrid(sample.grid.draw()))
            batch_decoder_inputs.append(input_seq_padded)
            batch_targets.append(target_seq_padded)

            # When batch is full, yield it
            if len(batch_encoder_inputs) == batch_size:
                # Stack the batch lists into arrays - Could probably do batched normalization/padding here.
                yield (
                    {
                        "encoder_inputs": np.stack(batch_encoder_inputs),
                        "decoder_inputs": np.stack(batch_decoder_inputs),
                    },
                    np.stack(batch_targets),
                )

                # Reset batch lists
                batch_encoder_inputs = []
                batch_decoder_inputs = []
                batch_targets = []

    def __new__(cls, sequence_length: int, batch_size: int):
        return tf.data.Dataset.from_generator(
            cls._generate,
            output_signature=(
                {
                    "encoder_inputs": tf.TensorSpec(
                        shape=(batch_size, 900), dtype=tf.int32
                    ),
                    "decoder_inputs": tf.TensorSpec(
                        shape=(batch_size, sequence_length), dtype=tf.int32
                    ),
                },
                tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int32),
            ),
            args=(sequence_length, batch_size),
        )
