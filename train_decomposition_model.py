import random
import sys
from datetime import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from arc_dsl import gridstr_coloured
from arc_dsl.models.decomposition.model import (
    DecompositionModel,
    create_model,
    sequence_length,
)
from arc_dsl.models.metrics import masked_accuracy, masked_loss
from arc_dsl.models.training_data import sample_multi_bitmap_grid
from arc_dsl.models.tokenizer import GridlispTokenizer
from arc_dsl.models.decomposition.training_data import DecompositionDataset
from arc_dsl.models.optimizers import get_attn_is_all_u_need_optimizer

# A name used for backups, checkpoints, and tensorboard logs
if len(sys.argv) < 2:
    training_session_name = datetime.now().strftime("%Y%m%d%H%M%S")  # Alpha-sortable
else:
    training_session_name = sys.argv[1]

backups_dir = "backups"
checkpoints_dir = "checkpoints"
tensorboard_log_dir = f"tensorboard_logs/{training_session_name}"
os.makedirs(tensorboard_log_dir, exist_ok=True)

tokenizer = GridlispTokenizer()

# Hyperparameters
embed_dim = 64
dense_dim = 2048
n_layers = 3
batch_size = 128
epochs = 5000  # We stop manually when we see loss flattened out
steps_per_epoch = 1000

# Create the model
transformer = create_model(embed_dim, dense_dim, n_layers, tokenizer)
transformer.summary()
transformer.compile(
    optimizer=get_attn_is_all_u_need_optimizer(dense_dim),
    loss=masked_loss,
    metrics=[masked_accuracy],
)

# Save the tokenizer for this run
tokenizer.save(
    f"{checkpoints_dir}/{transformer.name}_{training_session_name}-tokenizer.json"
)


class PrintPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer: GridlispTokenizer):
        super().__init__()
        rng = random.Random(42)

        self.tokenizer = tokenizer
        self.static_validation = [sample_multi_bitmap_grid(rng) for _ in range(3)]

    def on_test_end(self, logs=None):
        tf.print()

        inference_model = DecompositionModel(model=self.model, tokenizer=self.tokenizer)
        decompositions = inference_model.decompose_batch(
            [sample.grid.draw() for sample in self.static_validation]
        )

        for ix, sample in enumerate(self.static_validation):
            decomposition = decompositions[ix]
            expected_grid = sample.grid.draw()
            target_dsl = sample.dsl_sentence

            tf.print("=== Target-dsl:", target_dsl)

            if decomposition is None:
                tf.print("Decompostion failed")
                continue

            spacer = np.full((expected_grid.shape[0], 1), 5)
            stacked_grids = np.concatenate(
                [expected_grid, spacer, decomposition.draw()], axis=1
            )
            tf.print(gridstr_coloured(stacked_grids))


backup_callback = keras.callbacks.BackupAndRestore(
    f"{backups_dir}/{transformer.name}_{training_session_name}",
    save_freq="epoch",
    delete_checkpoint=False,
)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    f"{checkpoints_dir}/{transformer.name}_{training_session_name}.keras",
    save_freq="epoch",
    save_best_only=True,
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=tensorboard_log_dir,
    update_freq="epoch",
    histogram_freq=1,
    write_graph=False,
    write_steps_per_second=True,
)

# Infinite sequences of generated data
train_ds = DecompositionDataset(sequence_length, batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = DecompositionDataset(sequence_length, batch_size).prefetch(tf.data.AUTOTUNE)

# Fit the model
transformer.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=1,
    callbacks=[
        backup_callback,
        checkpoint_callback,
        PrintPredictionsCallback(tokenizer),
        tensorboard_callback,
    ],
)

print("Training done!")
