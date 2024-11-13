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
trained_models_dir = repo_root / "trained_models" / "pattern_completion"
model_name = "pattern_completion-v1"

sequence_length = 512


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
        name="pattern-completion-model",
    )
    model.supports_masking = True

    return model


def load_model():
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


class PatternCompletionModel:
    def __init__(self, model=None, tokenizer=None):
        self.model = load_model() if model is None else model
        self.tokenizer = load_tokenizer() if tokenizer is None else tokenizer

    def complete(
        self, grid: np.ndarray, paramd_dsl: str, sequence_length=sequence_length
    ) -> DeclarativeGrid:
        H, W = grid.shape
        encoder_inputs = tf.expand_dims(normgrid(grid), 0)
        tokenized_target_sentence = (
            [self.tokenizer.start_token_id]
            + self.tokenizer.encode(paramd_dsl)
            + [self.tokenizer.completion_token_id]
        )
        completion_ix = len(tokenized_target_sentence) - 1

        for i in range(completion_ix, sequence_length):
            padding = [0] * (sequence_length - len(tokenized_target_sentence))
            padded_input = tf.constant(tokenized_target_sentence + padding)

            logits = self.model.predict(
                {
                    "encoder_inputs": encoder_inputs,
                    "decoder_inputs": tf.expand_dims(padded_input, 0),
                },
                verbose=0,
            )

            next_token_logits = logits[0, i, :]
            sampled_token_index = np.argmax(next_token_logits)

            tokenized_target_sentence.append(sampled_token_index)

            if sampled_token_index == self.tokenizer.end_token_id:
                break

        if tokenized_target_sentence[-1] != self.tokenizer.end_token_id:
            logger.warning("Decomposition inference did not end with end token.")
            return None

        bmp_dsl = self.tokenizer.decode(
            tokenized_target_sentence[completion_ix + 1 : -1]
        )
        grid_dsl = f"(make-grid {H} {W} {bmp_dsl})"

        try:
            return eval_sentence(grid_dsl)
        except ParseError:
            logger.warning(f"Invalid syntax in bmp: {bmp_dsl}")
            return None
        except EvalError:
            logger.warning(f"Eval error in: {grid_dsl}")
            return None
