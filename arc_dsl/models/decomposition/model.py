import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from arc_dsl.errors import EvalError, ParseError
from arc_dsl.grid import DeclarativeGrid
from arc_dsl.interpreter import eval_sentence
from arc_dsl.models.layers import (
    PositionalEmbedding,
    PositionalGridEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)
from arc_dsl.models.metrics import masked_accuracy, masked_loss
from arc_dsl.models.training_data import normgrid
from arc_dsl.models.tokenizer import GridlispTokenizer

logger = logging.getLogger(__name__)

repo_root = Path(__file__).parent.parent.parent
trained_models_dir = repo_root / "trained_models" / "decomposition"
model_name = "decompostion-v3"

sequence_length = 256


def create_model(embed_dim, dense_dim, n_layers, tokenizer: GridlispTokenizer):
    dsl_vocab_size = len(tokenizer)
    grid_vocab_size = 11
    num_heads = 8

    encoder_inputs = keras.Input(shape=(900,), dtype="int64", name="encoder_inputs")
    encoder_outputs = PositionalGridEmbedding(grid_vocab_size, embed_dim)(
        encoder_inputs
    )

    for _ in range(n_layers):
        encoder_outputs = TransformerEncoderBlock(embed_dim, dense_dim, num_heads)(
            encoder_outputs
        )

    decoder_inputs = keras.Input(
        shape=(sequence_length,), dtype="int64", name="decoder_inputs"
    )

    inputs = PositionalEmbedding(sequence_length, dsl_vocab_size, embed_dim)(
        decoder_inputs
    )
    for _ in range(n_layers):
        inputs = TransformerDecoderBlock(embed_dim, dense_dim, num_heads)(
            [inputs, encoder_outputs]
        )

    inputs = layers.LayerNormalization()(inputs)
    decoder_outputs = layers.Dense(dsl_vocab_size)(inputs)

    model = keras.Model(
        inputs={"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
        outputs=decoder_outputs,
        name="decomposition-model",
    )
    model.supports_masking = True

    return model


def load_model():
    """Load the model used for decomposing ARC task input grids into objects."""
    model_path = trained_models_dir / f"{model_name}.keras"

    return keras.models.load_model(
        model_path,
        custom_objects={
            "PositionalEmbedding": PositionalEmbedding,
            "PositionalGridEmbedding": PositionalGridEmbedding,
            "TransformerEncoderBlock": TransformerEncoderBlock,
            "TransformerDecoderBlock": TransformerDecoderBlock,
            "masked_loss": masked_loss,
            "masked_accuracy": masked_accuracy,
        },
    )


def load_tokenizer():
    tokenizer_path = trained_models_dir / f"{model_name}-tokenizer.json"
    return GridlispTokenizer.load(tokenizer_path)


class DecompositionModel:
    def __init__(self, model=None, tokenizer=None):
        self.model = load_model() if model is None else model
        self.tokenizer = load_tokenizer() if tokenizer is None else tokenizer

    def decompose(
        self, grid: np.ndarray, sequence_length=sequence_length
    ) -> DeclarativeGrid:
        grids = np.expand_dims(grid, axis=0)
        results = self.decompose_batch(grids, sequence_length)
        return results[0]

    def decompose_batch(
        self, grids: list[np.ndarray], sequence_length=sequence_length
    ) -> list[DeclarativeGrid]:
        B = len(grids)
        grid_shapes = [grid.shape for grid in grids]
        sequences_done = np.zeros(B, dtype=bool)

        tokenized_target_sentences = np.zeros((B, sequence_length), dtype=int)
        tokenized_target_sentences[:, 0] = self.tokenizer.start_token_id
        grids = [normgrid(grid) for grid in grids]

        for i in range(sequence_length - 1):
            logits = self.model.predict(
                {
                    "encoder_inputs": tf.constant(grids),
                    "decoder_inputs": tf.constant(tokenized_target_sentences),
                },
                verbose=0,
            )

            next_token_logits = logits[:, i, :]
            sampled_token_indices = np.argmax(next_token_logits, axis=1)
            tokenized_target_sentences[:, i + 1] = sampled_token_indices

            sequences_done |= sampled_token_indices == self.tokenizer.end_token_id

            if np.all(sequences_done):
                break

        results = []
        for idx in range(B):
            tokenized_sentence = tokenized_target_sentences[idx]
            # Find the index of the end token
            end_token_indices = np.where(
                tokenized_sentence == self.tokenizer.end_token_id
            )[0]
            if len(end_token_indices) == 0:
                logger.warning(
                    f"Decomposition inference did not end with end token for sample {idx}."
                )
                results.append(None)
                continue
            else:
                end_idx = end_token_indices[0]

            # Exclude start token at position 0 and end token
            tokenized_sequence = tokenized_sentence[1:end_idx]
            bmp_dsl = self.tokenizer.decode(tokenized_sequence)
            H, W = grid_shapes[idx]
            grid_dsl = f"(make-grid {H} {W} {bmp_dsl})"

            try:
                result = eval_sentence(grid_dsl)
                results.append(result)
            except ParseError:
                logger.warning(f"Invalid syntax in bmp: {bmp_dsl} for sample {idx}")
                results.append(None)
            except EvalError:
                logger.warning(f"Eval error in: {grid_dsl} for sample {idx}")
                results.append(None)

        return results
