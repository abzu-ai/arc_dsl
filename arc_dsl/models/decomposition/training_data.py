import random
import numpy as np
import tensorflow as tf

from arc_dsl.models.tokenizer import GridlispTokenizer
from arc_dsl.models.training_data import normgrid, sample_multi_bitmap_grid


class DecompositionDataset(tf.data.Dataset):
    def _generate(sequence_length: int, batch_size:int):
        rng = random.Random()
        tokenizer = GridlispTokenizer()

        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_targets = []

        while True:
            sample = sample_multi_bitmap_grid(rng)
            dsl = "".join([bm.to_dsl() for bm in sample.grid.bitmaps])
            tokenized = (
                [tokenizer.start_token_id]
                + tokenizer.encode(dsl)
                + [tokenizer.end_token_id]
            )

            if len(tokenized) > sequence_length:
                continue # Too long

            if len(tokenized) < 5:
                continue # Too silly

            input_seq = tokenized[:-1]  # Strip [end] token, preserve [start] token
            target_seq = tokenized[1:]  # Strip [start] token, preserve [end] token
            padding_length = sequence_length - len(input_seq)

            input_seq_padded = np.pad(input_seq, (0, padding_length), 'constant')
            target_seq_padded = np.pad(target_seq, (0, padding_length), 'constant')

            # Collect samples into batches
            batch_encoder_inputs.append(normgrid(sample.grid.draw()))
            batch_decoder_inputs.append(input_seq_padded)
            batch_targets.append(target_seq_padded)

            # When batch is full, yield it
            if len(batch_encoder_inputs) == batch_size:
                # Stack the batch lists into arrays - Could probably do batched normalization/padding here.
                yield ({
                        "encoder_inputs": np.stack(batch_encoder_inputs),
                        "decoder_inputs": np.stack(batch_decoder_inputs),
                    },
                    np.stack(batch_targets)
                )

                # Reset batch lists
                batch_encoder_inputs = []
                batch_decoder_inputs = []
                batch_targets = []


    def __new__(cls, sequence_length:int, batch_size:int):
        return tf.data.Dataset.from_generator(
            cls._generate,
            output_signature=(
                {
                    "encoder_inputs": tf.TensorSpec(shape=(batch_size, 900), dtype=tf.int32),
                    "decoder_inputs": tf.TensorSpec(
                        shape=(batch_size, sequence_length), dtype=tf.int32
                    ),
                },
                tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int32),  # Output DSL
            ),
            args=(sequence_length, batch_size)
        )
